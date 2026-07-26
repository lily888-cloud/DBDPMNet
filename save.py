import os
def prepare_folders(args):
    # 定义保存路径（根据项目习惯，通常包含checkpoint和日志目录）
    args.save_path = os.path.join(args.save_path, args.store_name) if hasattr(args, 'save_path') else os.path.join('./checkpoint', args.store_name)
    args.log_path = os.path.join(args.log_path, args.store_name) if hasattr(args, 'log_path') else os.path.join('./logs', args.store_name)
    
    # 创建文件夹（如果不存在）
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
class AverageMeter(object):
    """
    计算并存储指标的当前值、平均值、总和及计数
    """
    def __init__(self, name, fmt=':f'):
        self.name = name  # 指标名称（如'Time'、'Loss'）
        self.fmt = fmt    # 格式化字符串（用于输出）
        self.reset()      # 初始化指标

    def reset(self):
        """重置所有指标为初始状态"""
        self.val = 0    # 当前值
        self.avg = 0    # 平均值
        self.sum = 0    # 总和
        self.count = 0  # 计数

    def update(self, val, n=1):
        """
        更新指标
        Args:
            val: 当前批次的指标值
            n: 批次大小（用于计算总和时加权）
        """
        self.val = val
        self.sum += val * n  # 累计总和（乘以批次大小，支持多样本平均）
        self.count += n      # 累计样本数
        self.avg = self.sum / self.count  # 计算平均值

    def __str__(self):
        """格式化输出字符串（当前值和平均值）"""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(** self.__dict__)