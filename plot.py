import matplotlib.pyplot as plt
import pandas as pd


effdet_his = pd.read_csv('./cache/efficientdet/train_his.csv')

plt.figure(figsize=(5, 2), dpi=300)
plt.plot(effdet_his['epoch'], effdet_his['train_loss'], label='train')
plt.plot(effdet_his['epoch'], effdet_his['val_loss'], label='val')
plt.title('Loss')
plt.ylim(0.35, 0.5)
plt.legend()
plt.savefig("effdet_train.png")
