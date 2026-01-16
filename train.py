
import config
import torch.nn as nn
from tqdm import tqdm
from instrument import train_disc
import torch
from utils import init_UNET, init_disc, init_dark, get_traindata, save_model, save_some_examples
from loss import Perceptual, Style, L1, BCE

def train(
    gen, opt_gen, scr_gen,
    disc, opt_disc, scr_disc,
    dark,
    loader, bce, l1, lp, ls
):
    loop = tqdm(loader, leave=True)
    for idx, (augmented_dust, augmented_clean, X, X_s, X_r, retinex_dust) in enumerate(loop):
        X_r = X_r.to(config.DEVICE)
        X = X.to(config.DEVICE)
        X_s = X_s.to(config.DEVICE)
        retinex_dust = retinex_dust.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake_s = gen(X_s)
            fake_r = gen(X_r)

        # calculating supervised loss (L1, Perceptual, and Style)
        L_sup = l1(fake_s, X) * config.LAMBDA_L1 + lp(X, fake_s) + ls(X, fake_s)
        
        disc, opt_disc, scr_disc, Loss_disc = train_disc(gen, disc, opt_disc, scr_disc, augmented_dust, augmented_clean, bce)

        L_adv = 0
        L_reg = 0
        L_psu = l1(retinex_dust, fake_r)*.1
        L_con = 0
        
        with torch.cuda.amp.autocast():
            for k in range(config.K_augmented):
                _X_r, noise = config.add_gaussian_noise_to_image(X_r)
                L_con += l1(fake_r, gen(_X_r))
                
                y = augmented_dust[k].to(config.DEVICE) # dusty images
                fake_y = gen(y)
                D_fake = disc(fake_y)
                L_adv += bce(D_fake, torch.ones_like(D_fake))
                fake_dark = dark(config.get_dc_tensor(fake_y))
                L_reg += bce(fake_dark, torch.ones_like(fake_dark))

        L_unsup = (L_con/config.K_augmented) + (L_reg/config.K_augmented) + (L_psu) + (L_adv/config.K_augmented)
                   

        Loss_gen = L_sup * config.LAMBDA_SUP + L_unsup * config.LAMBDA_UNSUP
        
        opt_gen.zero_grad()
        scr_gen.scale(Loss_gen).backward()
        scr_gen.step(opt_gen)
        scr_gen.update()

        loop.set_postfix(
            Loss_total = Loss_gen.item(),
            unsup = L_unsup.item(),
            sup = L_sup.item(),
        )

    return gen, opt_gen, scr_gen, disc, opt_disc, scr_disc



def main():
    gen, opt_gen, scr_gen = init_UNET()
    disc, opt_disc, scr_disc = init_disc()
    dark, _, _ = init_dark()
    dark.eval()
    #-------------------------------------------------------------------------------------------------
    loader = get_traindata()
    #-------------------------------------------------------------------------------------------------    
    lp = Perceptual()
    ls = Style()
    l1 = L1()
    bce = BCE()
    #-------------------------------------------------------------------------------------------------
    # train models
    for epoch in range(config.NUM_EPOCHS):
        gen, opt_gen, scr_gen, disc, opt_disc, scr_disc = train(
            gen, opt_gen, scr_gen, disc, opt_disc, scr_disc, dark, loader, bce, l1, lp, ls
        )
        save_model(gen, opt_gen, config.UNIT_PATH)
        save_model(disc, opt_disc, config.DISCRIMINATOR_PATH)
    
    # save final model and outputs
    # save_model(gen, opt_gen, config.UNIT_PATH)
    # save_model(disc, opt_disc, config.DISCRIMINATOR_PATH)
    # save_some_examples(gen, loader, config.NUM_EPOCHS, "outcomes/gen")

if __name__ == "__main__":
    main()