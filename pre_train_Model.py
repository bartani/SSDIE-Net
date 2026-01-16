
import config
import torch.nn as nn
from tqdm import tqdm
from loss import Perceptual, Style, L1
import torch
from utils import init_UNET, get_pre_traindata, get_testdata, save_some_examples, save_model

def train(
    gen, opt_gen, scr_gen,
    loader, l1, ls, lp
):
    loop = tqdm(loader, leave=True)
    for idx, (X, X_s) in enumerate(loop):
        X = X.to(config.DEVICE)
        X_s = X_s.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(X_s)
        style_loss = ls(X, fake)
        perceptual_loss = lp(X, fake)
        l1_loss = l1(X, fake)
        G_loss = (
            style_loss
            + perceptual_loss
            + l1_loss * config.LAMBDA_L1
        )
        
        opt_gen.zero_grad()
        scr_gen.scale(G_loss).backward()
        scr_gen.step(opt_gen)
        scr_gen.update()

        loop.set_postfix(
            style_loss = style_loss.item(),
            perceptual_loss = perceptual_loss.item(),
            l1_loss = l1_loss.item(),
            Loss_gen = G_loss.item(),
        )

    return gen, opt_gen, scr_gen

def main():
    gen, opt_gen, scr_gen = init_UNET()
    #-------------------------------------------------------------------------------------------------
    loader = get_pre_traindata()
    test_loader = get_testdata()
    #-------------------------------------------------------------------------------------------------    
    lp = Perceptual()
    ls = Style()
    l1 = L1()
    #-------------------------------------------------------------------------------------------------
    # train models
    for epoch in range(config.NUM_EPOCHS_PRE_TRAIN):
        gen, opt_gen, scr_gen = train(
            gen, opt_gen, scr_gen, loader, l1, ls, lp
        )
        # if you want to save checkpoints per-epoch, uncomment bellow code
        # save_model(gen, opt_gen)
    
    # # save final model and outputs
    save_model(gen, opt_gen, config.UNIT_PATH)
    save_some_examples(gen, test_loader, config.NUM_EPOCHS_PRE_TRAIN, "outcomes/Pre_Train")

if __name__ == "__main__":
    main()