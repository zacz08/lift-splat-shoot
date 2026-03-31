import re
import os
import pandas as pd
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_log_and_plot(log_file_path):
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered_lines = [line for line in lines if "train/loss_epoch=" in line]

    rows = []
    for line in filtered_lines:
        # get epoch number
        epoch_match = re.search(r"Epoch (\d+):", line)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))
        if epoch == 0:
            continue

        # train loss
        train_simple = re.search(r"train/loss_simple_epoch=([\d.]+)", line)
        train_vlb = re.search(r"train/loss_vlb_epoch=([\d.]+)", line)
        train_seg = re.search(r"train/loss_seg_epoch=([\d.]+)", line)
        train_c = re.search(r"train/loss_c=([\d.]+)", line)
        train_o = re.search(r"train/loss_o=([\d.]+)", line)
        train_total = re.search(r"train/loss_epoch=([\d.]+)", line)

        # val loss
        val_simple = re.search(r"val/loss_simple=([\d.]+)", line)
        val_vlb = re.search(r"val/loss_vlb=([\d.]+)", line)
        val_seg = re.search(r"val/loss_seg=([\d.]+)", line)
        val_c = re.search(r"val/loss_c=([\d.]+)", line)
        val_o = re.search(r"val/loss_o=([\d.]+)", line)
        val_total = re.search(r"val/loss=([\d.]+)", line)

        row = {
            "epoch": epoch,
            "loss_simple_epoch": float(train_simple.group(1)) if train_simple else None,
            "loss_vlb_epoch": float(train_vlb.group(1)) if train_vlb else None,
            "loss_seg_epoch": float(train_seg.group(1)) if train_seg else None,
            "loss_c": float(train_c.group(1)) if train_c else None,
            "loss_o": float(train_o.group(1)) if train_o else None,
            "loss_epoch": float(train_total.group(1)) if train_total else None,
            "val_loss_simple": float(val_simple.group(1)) if val_simple else None,
            "val_loss_vlb": float(val_vlb.group(1)) if val_vlb else None,
            "val_loss_seg": float(val_seg.group(1)) if val_seg else None,
            "val_loss_c": float(val_c.group(1)) if val_c else None,
            "val_loss_o": float(val_o.group(1)) if val_o else None,
            "val_loss": float(val_total.group(1)) if val_total else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by="epoch")
    df = df.drop_duplicates(subset="epoch", keep="last")

    plt.figure(figsize=(12, 7))

    # train loss
    plt.plot(df["epoch"], df["loss_simple_epoch"], label="train/loss_simple_epoch")
    plt.plot(df["epoch"], df["loss_epoch"], label="train/loss_epoch")
    if "loss_vlb_epoch" in df.columns and df["loss_vlb_epoch"].notna().any():
        plt.plot(df["epoch"], df["loss_vlb_epoch"], label="train/loss_vlb_epoch")
    if "loss_seg_epoch" in df.columns and df["loss_seg_epoch"].notna().any():
        plt.plot(df["epoch"], df["loss_seg_epoch"], label="train/loss_seg_epoch")
    if "loss_c_epoch" in df.columns and df["loss_c_epoch"].notna().any():
        plt.plot(df["epoch"], df["loss_c_epoch"], label="train/loss_c_epoch")
    if "loss_o_epoch" in df.columns and df["loss_o_epoch"].notna().any():
        plt.plot(df["epoch"], df["loss_o_epoch"], label="train/loss_o_epoch")

    # val loss
    if "val_loss_simple" in df.columns and df["val_loss_simple"].notna().any():
        plt.plot(df["epoch"], df["val_loss_simple"], '--', label="val/loss_simple")
    if "val_loss_vlb" in df.columns and df["val_loss_vlb"].notna().any():
        plt.plot(df["epoch"], df["val_loss_vlb"], '--', label="val/loss_vlb")
    if "val_loss_seg" in df.columns and df["val_loss_seg"].notna().any():
        plt.plot(df["epoch"], df["val_loss_seg"], '--', label="val/loss_seg")
    if "val_loss_c" in df.columns and df["val_loss_c"].notna().any():
        plt.plot(df["epoch"], df["val_loss_c"], '--', label="val/loss_c")
    if "val_loss_o" in df.columns and df["val_loss_o"].notna().any():
        plt.plot(df["epoch"], df["val_loss_o"], '--', label="val/loss_o")
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        plt.plot(df["epoch"], df["val_loss"], '--', label="val/loss")

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training & Validation Loss vs. Epoch")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def parse_csv_and_plot(csv_path, output_path, fields_to_plot=None):
    """
    Parse CSV and plot training curves.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the output plot
        fields_to_plot: List of field names to plot. If None, plots default fields.
                       Fields should be in format like "train/loss_epoch", "val/IoU", etc.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    df = df.sort_values(by="epoch").drop_duplicates("epoch")

    plt.figure(figsize=(12, 7))

    def safe_plot(column, label, style='-'):
        if column in df.columns and df[column].notna().any():
            # 转换为 numpy array 以避免 pandas 多维索引错误
            plt.plot(df["epoch"].values, df[column].values, style, label=label)

    # 如果没有指定字段，使用默认字段
    if fields_to_plot is None:
        fields_to_plot = [
            "train/loss_seg_epoch",
            "val/loss_seg",
            "val/IoU"
        ]
    
    # 根据字段列表绘制曲线
    for field in fields_to_plot:
        # 确定线型：验证集使用虚线，训练集使用实线
        style = '--' if field.startswith('val/') else '-'
        # 提取简洁的标签名
        label = field
        safe_plot(field, label, style)

    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.title("Training & Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parse_log_and_plot("./tools/training_log.log")
