
import config
import torch.nn as nn
from tqdm import tqdm
from loss import BCE
import torch
from utils import init_dark, get_train_darkchannel_data, save_some_examples, save_model

def train(
    model, opt, scr,
    loader, bce, epoch
):
    loop = tqdm(loader, leave=True)
    for idx, (X, X_r) in enumerate(loop):
        X = X.to(config.DEVICE)
        X_r = X_r.to(config.DEVICE)

        _X = config.get_dc_tensor(X)
        _X_r = config.get_dc_tensor(X_r)
        with torch.cuda.amp.autocast():
            fake_x = model(_X)
            fake_xr = model(_X_r)

        _loss = (bce(fake_x, torch.ones_like(fake_x)) + bce(fake_xr, torch.zeros_like(fake_xr)))*.5
                
        opt.zero_grad()
        scr.scale(_loss).backward()
        scr.step(opt)
        scr.update()

        loop.set_postfix(
            epoch = epoch,
            Loss_model = _loss.item(),
        )

    return model, opt, scr

def main():
    model, opt, scr = init_dark()
    #-------------------------------------------------------------------------------------------------
    loader = get_train_darkchannel_data()
    #-------------------------------------------------------------------------------------------------    
    bce = BCE()
    #-------------------------------------------------------------------------------------------------
    # train models
    for epoch in range(config.NUM_EPOCHS_PRE_TRAIN):
        model, opt, scr = train(
            model, opt, scr, loader, bce, epoch
        )
        # if you want to save checkpoints per-epoch, uncomment bellow code
        save_model(model, opt, config.DARKCHANNEL_PATH)
    
    # # save final model and outputs
    save_model(model, opt, config.DARKCHANNEL_PATH)
    # save_some_examples(gen, test_loader, config.NUM_EPOCHS_PRE_TRAIN, "outcomes/Pre_Train")

if __name__ == "__main__":
    main()