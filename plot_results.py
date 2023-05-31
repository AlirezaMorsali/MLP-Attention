import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


experiment_name = "Final"
result_path = f"results/{experiment_name}"


with open(f"{result_path}/hyper_params.json") as f:
    data = json.load(f)

start_index = 2


data = {"epoch": data["epochs"][start_index:],
        "standard training loss": data["train_loss"][start_index:],
        "standard validation loss": data["val_loss"][start_index:],
        "mlp training loss": data["mlp_attention_train_loss"][start_index:],
        "mlp validation loss": data["mlp_attention_val_loss"][start_index:],
        }

df = pd.DataFrame(data)
sns.set(font_scale=2)
plt.figure(figsize=(12, 8))
# plot the data using seaborn
sns.set_style("darkgrid")
sns.lineplot(x="epoch", y="standard training loss", data=df, label="training loss (standard transformer)")
sns.lineplot(x="epoch", y="standard validation loss", data=df,
             label="validation loss (standard transformer)", linestyle="--")

sns.lineplot(x="epoch", y="mlp training loss", data=df, label="training loss (mlp-attention)")
sns.lineplot(x="epoch", y="mlp validation loss", data=df, label="validation loss (mlp-attention)", linestyle="--")


# set the axis labels and title
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('training and validation losses')

# display the legend
plt.legend()

plt.savefig('results/mlpvstrans.png', dpi=500)

# display the plot
# plt.show()
