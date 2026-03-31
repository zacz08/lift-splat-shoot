import os
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ModelSummary(Callback):
    """
    A PyTorch Lightning callback to log model parameter statistics.
    Saves detailed model summary including total parameters, trainable parameters,
    and parameters per module to a text file.
    """
    
    def __init__(self, log_folder=None):
        super().__init__()
        self.log_folder = log_folder
        
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """Called when training begins."""
        if self.log_folder is None:
            self.log_folder = trainer.log_dir if hasattr(trainer, 'log_dir') else './logs'
        
        os.makedirs(self.log_folder, exist_ok=True)
        summary_path = os.path.join(self.log_folder, 'model_summary.txt')
        
        # Generate and save model summary
        summary_text = self.generate_model_summary(pl_module)
        
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        print(f"[ModelSummary] Saved to {summary_path}")
        # print("\n" + "="*80)
        print(summary_text)
        # print("="*80 + "\n")
    
    def generate_model_summary(self, model):
        """Generate detailed model summary."""
        lines = []
        lines.append("="*80)
        lines.append("MODEL SUMMARY")
        lines.append("="*80)
        lines.append(f"Model: {model.__class__.__name__}")
        lines.append("")
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        lines.append("-" * 80)
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Total Parameters:        {total_params:,}")
        lines.append(f"Trainable Parameters:    {trainable_params:,}")
        lines.append(f"Non-trainable Parameters: {non_trainable_params:,}")
        lines.append(f"Model Size (MB):         {total_params * 4 / (1024**2):.2f}")
        lines.append("")
        
        # # Get module-wise statistics
        # lines.append("-" * 80)
        # lines.append("MODULE-WISE STATISTICS")
        # lines.append("-" * 80)
        # lines.append(f"{'Module Name':<40} {'Parameters':>15} {'Trainable':>15}")
        # lines.append("-" * 80)
        
        # module_stats = self.get_module_stats(model)
        # for name, (params, trainable) in sorted(module_stats.items()):
        #     trainable_str = f"{trainable:,}" if trainable > 0 else "-"
        #     lines.append(f"{name:<40} {params:>15,} {trainable_str:>15}")
        
        # lines.append("-" * 80)
        # lines.append("")
        
        # # Detailed layer-by-layer breakdown
        # lines.append("-" * 80)
        # lines.append("DETAILED LAYER BREAKDOWN")
        # lines.append("-" * 80)
        # lines.append(f"{'Layer Name':<50} {'Shape':<25} {'Parameters':>15}")
        # lines.append("-" * 80)
        
        # for name, param in model.named_parameters():
        #     shape_str = 'x'.join(map(str, param.shape))
        #     lines.append(f"{name:<50} {shape_str:<25} {param.numel():>15,}")
        
        # lines.append("-" * 80)
        # lines.append("")
        
        # # Configuration summary
        # if hasattr(model, 'cfg'):
        #     lines.append("-" * 80)
        #     lines.append("MODEL CONFIGURATION")
        #     lines.append("-" * 80)
            
        #     cfg = model.cfg
            
        #     if hasattr(cfg, 'model'):
        #         lines.append("Model Config:")
        #         lines.append(f"  downsample: {cfg.model.downsample}")
        #         lines.append(f"  camC: {cfg.model.camC}")
        #         lines.append(f"  outC: {cfg.model.outC}")
        #         lines.append("")
            
        #     if hasattr(cfg, 'grid_conf'):
        #         lines.append("Grid Config:")
        #         lines.append(f"  xbound: {cfg.grid_conf.xbound}")
        #         lines.append(f"  ybound: {cfg.grid_conf.ybound}")
        #         lines.append(f"  zbound: {cfg.grid_conf.zbound}")
        #         lines.append(f"  dbound: {cfg.grid_conf.dbound}")
        #         lines.append("")
            
        #     if hasattr(cfg, 'data_aug'):
        #         lines.append("Data Augmentation Config:")
        #         lines.append(f"  H: {cfg.data_aug.H}")
        #         lines.append(f"  W: {cfg.data_aug.W}")
        #         lines.append(f"  final_dim: {cfg.data_aug.final_dim}")
        #         lines.append(f"  resize_lim: {cfg.data_aug.resize_lim}")
        #         lines.append("")
            
        #     lines.append("-" * 80)
        #     lines.append("")
        
        # Architecture overview
        lines.append("-" * 80)
        lines.append("ARCHITECTURE OVERVIEW")
        lines.append("-" * 80)
        
        if hasattr(model, 'camencode'):
            cam_params = sum(p.numel() for p in model.camencode.parameters())
            cam_trainable = sum(p.numel() for p in model.camencode.parameters() if p.requires_grad)
            lines.append(f"Camera Encoder (CamEncode):")
            lines.append(f"  Total params: {cam_params:,}")
            lines.append(f"  Trainable:    {cam_trainable:,}")
            lines.append("")
        
        if hasattr(model, 'bevencode'):
            bev_params = sum(p.numel() for p in model.bevencode.parameters())
            bev_trainable = sum(p.numel() for p in model.bevencode.parameters() if p.requires_grad)
            lines.append(f"BEV Encoder (BevEncode):")
            lines.append(f"  Total params: {bev_params:,}")
            lines.append(f"  Trainable:    {bev_trainable:,}")
            lines.append("")
        
        lines.append("-" * 80)
        lines.append("")
        
        return '\n'.join(lines)
    
    def get_module_stats(self, model):
        """Get parameter statistics for each module."""
        module_stats = {}
        
        for name, module in model.named_children():
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if total_params > 0:
                module_stats[name] = (total_params, trainable_params)
            
            # Recursively get sub-module stats
            sub_stats = self._get_submodule_stats(module, prefix=name)
            module_stats.update(sub_stats)
        
        return module_stats
    
    def _get_submodule_stats(self, module, prefix=''):
        """Recursively get statistics for sub-modules."""
        stats = {}
        
        for name, submodule in module.named_children():
            full_name = f"{prefix}.{name}"
            
            # Only include leaf modules or important containers
            children = list(submodule.children())
            if len(children) == 0 or isinstance(submodule, (torch.nn.Sequential, torch.nn.ModuleList)):
                total_params = sum(p.numel() for p in submodule.parameters())
                trainable_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
                
                if total_params > 0:
                    stats[full_name] = (total_params, trainable_params)
            
            # Recurse for nested modules (but limit depth)
            if len(children) > 0 and prefix.count('.') < 2:
                sub_stats = self._get_submodule_stats(submodule, prefix=full_name)
                stats.update(sub_stats)
        
        return stats
    
    @staticmethod
    def count_parameters(model):
        """Utility function to count model parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
