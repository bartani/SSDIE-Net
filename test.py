from utils import init_UNET, init_PTC_GEN, get_testdata
from tqdm import tqdm
import config
import torch
from torchvision.utils import save_image

def test():
    gen, _, _ = init_UNET()
    gen.eval()

    loader = get_testdata()
    loop = tqdm(loader, leave=True)
    for idx, (X_r, _) in enumerate(loop):
        X_r = X_r.to(config.DEVICE)
        with torch.no_grad():
            fake = gen(X_r)
            
            can = torch.cat([X_r*.5+.5, fake*.5+.5], 0)
            save_image(can, f"{config.ENHANCED_OUTPUT_PATH}/dest_free_ptc_{idx}.png")

if __name__ == "__main__":

    test()
