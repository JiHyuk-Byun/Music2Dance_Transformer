# RUN CODE

```python
# Run code
CUDA_VISIBLE_DEVICES=GPU_NUMBER \
HYDRA_FULL_ERROR=1 \
python3 train.py \
name="Aist++_M2D_reconstruction_onlymse" \
server_name=SERVER_NAME \
gpu_name=GPU_NAME \
experiment=AISTPP \
trainer.min_epochs=1 \
trainer.max_epochs=6000 \
datamodule.batch_size=16 \
callbacks.model_checkpoint.every_n_train_steps=100000 \
model.gen_params.reconstruction=True \
model.loss_params.alphas=[0.636,2.964]
```