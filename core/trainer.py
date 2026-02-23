import os
import glob
import logging
import importlib
from tqdm import tqdm
from distutils.version import LooseVersion
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from core.adv_loss import AdversarialLoss
from core.dataset import TrainDataset
from model.modules.flow_comp import FlowCompletionLoss
from model.modules.disp_est import MonodepthLoss
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.num_local_frames = config['train_data_loader']['num_local_frames']
        self.num_ref_frames = config['train_data_loader']['num_ref_frames']
        self.spynet_lr = config['trainer'].get('spynet_lr', 1.0)

        # setup data set and data loader
        self.train_dataset = TrainDataset(config['train_data_loader'])

        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)

        # set loss functions
        self.adversarial_loss = AdversarialLoss(
            type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()
        self.flow_comp_loss = FlowCompletionLoss().to(self.config['device'])
        
        self.mono_loss = MonodepthLoss().to(self.config['device'])

        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])
        self.netG = net.InpaintGenerator()
        
        pretrained_gen_model = None

        if pretrained_gen_model is not None:
            state = torch.load(pretrained_gen_model)
            newdict = {}
            for k, v in state.items():
                if k in self.netG.state_dict().keys():
                    newdict[k] = v
            self.netG.load_state_dict(newdict, strict=False)
        
        print(self.netG)
        self.netG = self.netG.to(self.config['device'])
        if not self.config['model']['no_dis']:
            self.netD = net.Discriminator(
                in_channels=3,
                use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            self.netD = self.netD.to(self.config['device'])

        pretrained_dis_model = None

        if pretrained_dis_model is not None:
            state = torch.load(pretrained_dis_model)
            newdict = {}
            for k, v in state.items():
                if k in self.netD.state_dict().keys():
                    newdict[k] = v
            self.netD.load_state_dict(newdict, strict=False)

        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        if config['distributed']:
            self.netG = DDP(self.netG,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=True,
                            find_unused_parameters=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(self.netD,
                                device_ids=[self.config['local_rank']],
                                output_device=self.config['local_rank'],
                                broadcast_buffers=True,
                                find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        spynet_params = []
        for name, param in self.netG.named_parameters():
            if 'update_spynet' in name:
                spynet_params.append(param)
            else:
                backbone_params.append(param)

        optim_params = [
            {
                'params': backbone_params,
                'lr': self.config['trainer']['lr']
            },
            {  # finetuning learning rate for spynet
                'params': spynet_params,
                'lr': self.config['trainer']['lr'] * self.spynet_lr
            },
        ]

        self.optimG = torch.optim.Adam(optim_params,
                                       betas=(self.config['trainer']['beta1'],
                                              self.config['trainer']['beta2']))

        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(
                self.netD.parameters(),
                lr=self.config['trainer']['lr'],
                betas=(self.config['trainer']['beta1'],
                       self.config['trainer']['beta2']))

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(
                self.optimG,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
            self.scheD = MultiStepRestartLR(
                self.optimD,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
            self.scheD = CosineAnnealingRestartLR(
                self.optimD,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()
        self.scheD.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimG.param_groups[0]['lr']

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'),
                                'r').read().splitlines()[-1]
        else:
            ckpts = [
                os.path.basename(i).split('.pth')[0]
                for i in glob.glob(os.path.join(model_path, '*.pth'))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path,
                                    f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path,
                                    f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path,
                                    f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)
            if not self.config['model']['no_dis']:
                dataD = torch.load(dis_path,
                                   map_location=self.config['device'])
                self.netD.load_state_dict(dataD)

            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])
            self.scheG.load_state_dict(data_opt['scheG'])
            if not self.config['model']['no_dis']:
                self.optimD.load_state_dict(data_opt['optimD'])
                self.scheD.load_state_dict(data_opt['scheD'])
            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']

        else:
            if self.config['global_rank'] == 0:
                print('Warnning: There is no trained model found.'
                      'An initialized model will be used.')


    def get_key_ref_index(self, f, neighbor_ids, ref_frame_num, masks):
        ref_index = []
        current_mask = masks[0,f,0,:,:]
        ref_frames = list(set(list(range(f+1))) - set(neighbor_ids)) 

        if ref_frame_num == 0:
            ref_index = []
            return ref_index

        if len(ref_frames) <= ref_frame_num:
            for r in ref_frames:
                ref_index.append(r)
        elif len(ref_frames) > ref_frame_num and len(ref_frames) <= 5*ref_frame_num:
            ref_length = len(ref_frames) // ref_frame_num
            ref_seg_all = [ref_frames[i : i + ref_length] for i in range(len(ref_frames)-ref_frame_num*ref_length, len(ref_frames), ref_length)]
            for seg_index in range(len(ref_seg_all)):
                ref_index.append(ref_seg_all[seg_index][-1])
        elif len(ref_frames) > 5*ref_frame_num:
            ref_frames = ref_frames[-5*ref_frame_num:]
            ref_length = len(ref_frames) // ref_frame_num
            ref_seg_all = [ref_frames[i : i + ref_length] for i in range(len(ref_frames)-ref_frame_num*ref_length, len(ref_frames), ref_length)]
            for seg_index in range(len(ref_seg_all)):
                ref_index.append(ref_seg_all[seg_index][-1])

        return ref_index


    def save(self, it):
        
        # save model
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'],
                                    f'gen_{it:06d}.pth')
            dis_path = os.path.join(self.config['save_dir'],
                                    f'dis_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'],
                                    f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) \
               or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            if not self.config['model']['no_dis']:
                torch.save(netD.state_dict(), dis_path)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict(),
                        'scheG': self.scheG.state_dict(),
                        'scheD': self.scheD.state_dict()
                    }, opt_path)
            else:
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'scheG': self.scheG.state_dict()
                    }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar,
                        initial=self.iteration,
                        dynamic_ncols=True,
                        smoothing=0.01)

        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')

        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    def _train_epoch(self, pbar):
        """Process input and calculate loss every training epoch"""
        device = self.config['device']

        for frames, masks, _ in self.train_loader:
            self.iteration += 1

            frames, masks = frames.to(device), masks.to(device)  # frames -1~1 masks 0-1
            b, tx2, c, h, w = frames.size()
            t = int(tx2//2)
            
            l_t = self.num_local_frames

            masked_frames = (frames * (1 - masks).float())
            
            frames_left = frames[:,:t,:,:,:]
            frames_right = frames[:,t:,:,:,:]
            gt_local_frames_left = (frames_left[:, :l_t, ...] + 1) / 2
            gt_local_frames_right = (frames_right[:, :l_t, ...] + 1) / 2

            pred_imgs, pred_flows, pred_disps = self.netG(masked_frames, l_t)
            pred_imgs = pred_imgs.view(b, -1, c, h, w)
            comp_imgs = frames * (1. - masks) + masks * pred_imgs

            pred_flows_left = pred_flows[:2]
            pred_flows_right = pred_flows[2:]
            # compute flow completion loss
            flow_loss_left = self.flow_comp_loss(pred_flows_left, gt_local_frames_left)
            flow_loss_right = self.flow_comp_loss(pred_flows_right, gt_local_frames_right)
            flow_loss = flow_loss_left + flow_loss_right
            
            # compute disparity loss：
            gt_frames_stereo = torch.cat((frames[:,:t,:,:,:], frames[:,t:,:,:,:]), dim=2).view(-1, c*2, h, w)
            gt_frames_stereo = (gt_frames_stereo + 1) / 2 
            gt_frames_stereo = F.interpolate(gt_frames_stereo.view(-1, c, h, w),
                                            size=(192,384),
                                            mode='bilinear',
                                            align_corners=True)
            gt_frames_stereo = gt_frames_stereo.view(-1, c*2, 192, 384)
            gt_pred_disps = self.netG.forward_disp_pred(gt_frames_stereo)
            gt_disps = gt_pred_disps.view(-1, 2, 192,384)
            self.l1_criterion = nn.L1Loss()
            disp_est_loss = self.l1_criterion(pred_disps, gt_disps.detach())
            
            # monodepth loss gt
            left = frames[:,:t,:,:,:].view(-1, c, h, w)
            right = frames[:,t:,:,:,:].view(-1, c, h, w)   
            left = (left+1)/2 
            right = (right+1)/2
            mono_loss_gt, image_loss_gt, disp_gradient_loss_gt, lr_loss_gt = self.mono_loss([gt_disps], [left, right])

            # monodepth loss pred
            pred_left = pred_imgs[:,:t,:,:,:].view(-1, c, h, w)
            pred_right = pred_imgs[:,t:,:,:,:].view(-1, c, h, w)
            pred_left = (pred_left+1)/2 
            pred_right = (pred_right+1)/2                
            mono_loss_pred, image_loss_pred, disp_gradient_loss_pred, lr_loss_pred = self.mono_loss([pred_disps], [pred_left, pred_right])

            # Swap
            mono_loss, image_loss, disp_gradient_loss, lr_loss = self.mono_loss([pred_disps], [left, right])
            mono_loss_2, image_loss_2, disp_gradient_loss_2, lr_loss_2 = self.mono_loss([gt_disps], [pred_left, pred_right])
            
            gen_loss = 0
            dis_loss = 0

            if not self.config['model']['no_dis']:
                # discriminator adversarial loss
                real_clip = self.netD(frames)
                fake_clip = self.netD(comp_imgs.detach())
                dis_real_loss = self.adversarial_loss(real_clip, True, True)
                dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2
                self.add_summary(self.dis_writer, 'loss/dis_vid_fake',
                                 dis_fake_loss.item())
                self.add_summary(self.dis_writer, 'loss/dis_vid_real',
                                 dis_real_loss.item())
                self.optimD.zero_grad()
                dis_loss.backward()
                self.optimD.step()

                # generator adversarial loss
                gen_clip = self.netD(comp_imgs)
                gan_loss = self.adversarial_loss(gen_clip, True, False)
                gan_loss = gan_loss \
                    * self.config['losses']['adversarial_weight']
                gen_loss += gan_loss
                self.add_summary(self.gen_writer, 'loss/gan_loss',
                                 gan_loss.item())

            flow_loss = flow_loss * self.config['losses']['flow_weight']
            gen_loss += flow_loss
            self.add_summary(self.gen_writer, 'loss/flow_loss',
                             flow_loss.item())

            gen_loss += disp_est_loss
            self.add_summary(self.gen_writer, 'loss/disp_est_loss',
                             disp_est_loss.item())

            gen_loss += self.config['losses']['mono_weight'] * mono_loss
            self.add_summary(self.gen_writer, 'loss/mono_loss',
                             mono_loss.item())
            self.add_summary(self.gen_writer, 'loss/image_loss',
                             image_loss.item())
            self.add_summary(self.gen_writer, 'loss/disp_gradient_loss',
                             disp_gradient_loss.item())
            self.add_summary(self.gen_writer, 'loss/lr_loss',
                             lr_loss.item())

            gen_loss += self.config['losses']['mono_weight'] * mono_loss_2
            self.add_summary(self.gen_writer, 'loss/mono_loss_2',
                             mono_loss_2.item())
            self.add_summary(self.gen_writer, 'loss/image_loss_2',
                             image_loss_2.item())
            self.add_summary(self.gen_writer, 'loss/disp_gradient_loss_2',
                             disp_gradient_loss_2.item())
            self.add_summary(self.gen_writer, 'loss/lr_loss_2',
                             lr_loss_2.item())

            gen_loss += self.config['losses']['mono_weight'] * mono_loss_gt
            self.add_summary(self.gen_writer, 'loss/mono_loss_gt',
                             mono_loss_gt.item())
            self.add_summary(self.gen_writer, 'loss/image_loss_gt',
                             image_loss_gt.item())
            self.add_summary(self.gen_writer, 'loss/disp_gradient_loss_gt',
                             disp_gradient_loss_gt.item())
            self.add_summary(self.gen_writer, 'loss/lr_loss_gt',
                             lr_loss_gt.item())

            gen_loss += self.config['losses']['mono_weight'] * mono_loss_pred
            self.add_summary(self.gen_writer, 'loss/mono_loss_pred',
                             mono_loss_pred.item())
            self.add_summary(self.gen_writer, 'loss/image_loss_pred',
                             image_loss_pred.item())
            self.add_summary(self.gen_writer, 'loss/disp_gradient_loss_pred',
                             disp_gradient_loss_pred.item()) 
            self.add_summary(self.gen_writer, 'loss/lr_loss_pred',
                             lr_loss_pred.item())

            hole_loss = self.l1_loss(pred_imgs * masks, frames * masks)
            hole_loss = hole_loss / torch.mean(masks) \
                * self.config['losses']['hole_weight']
            gen_loss += hole_loss
            self.add_summary(self.gen_writer, 'loss/hole_loss',
                             hole_loss.item())

            valid_loss = self.l1_loss(pred_imgs * (1 - masks),
                                      frames * (1 - masks))
            valid_loss = valid_loss / torch.mean(1-masks) \
                * self.config['losses']['valid_weight']
            gen_loss += valid_loss
            self.add_summary(self.gen_writer, 'loss/valid_loss',
                             valid_loss.item())

            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            self.update_learning_rate()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                if not self.config['model']['no_dis']:
                    pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                          f"d: {dis_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}; "
                                          f"est: {disp_est_loss.item():.6f}; "
                                          f"mono1: {mono_loss.item():.3f}; "
                                          f"mono2: {mono_loss_2.item():.3f}"
                                          ))
                else:
                    pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))

                if self.iteration % self.train_args['log_freq'] == 0:
                    if not self.config['model']['no_dis']:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"flow: {flow_loss.item():.4f}; "
                                     f"d: {dis_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}; "
                                     f"est: {disp_est_loss.item():.6f}; "
                                     f"mono1: {mono_loss.item():.3f}; "
                                     f"mono2: {mono_loss_2.item():.3f}"
                                     )
                    else:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"flow: {flow_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")

            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args['iterations']:
                break
