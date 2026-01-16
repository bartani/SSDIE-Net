import torch
import config



def train_disc(gen, disc, opt, scr, augmented_dust, augmented_clean, bce):
    # Train Discriminator
    LOSS = 0
    
    with torch.cuda.amp.autocast():
        for k in range(config.K_augmented):
            x = augmented_clean[k].to(config.DEVICE) # clean images
            y = augmented_dust[k].to(config.DEVICE) # dusty images
            D_real = disc(x)
            D_fake = disc(gen(y))
            loss = (bce(D_real, torch.ones_like(D_real)) + bce(D_fake, torch.zeros_like(D_fake)))*.5
            LOSS+=loss
        LOSS = LOSS/config.K_augmented
        
        opt.zero_grad(); scr.scale(LOSS).backward(); scr.step(opt); scr.update()
    
    return disc, opt, scr, LOSS

def train_dark(dark, opt, scr, fake, real, bce):
    # Train Discriminator
    with torch.cuda.amp.autocast():
        D_real = dark(real)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = dark(fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        LOSS = (D_real_loss + D_fake_loss) / 2
    
    opt.zero_grad()
    scr.scale(LOSS).backward()
    scr.step(opt)
    scr.update()
    
    return dark, opt, scr, LOSS
