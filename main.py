from models.components.model import *

if __name__ == '__main__':
    ####################################################
    print("[*] Start DanceGeneratorAMNG_D")
    model = M2D(predict_length=30)

    audio = torch.randn(2, 240, 441)
    audioF = torch.randn(2, 60 * 11 + 90, 441)

    noise = torch.randn(2, 256)

    motion = torch.randn(2, 120, 24 * 3 + 3)
  
    genre = torch.tensor([3, 8])

    pred_motion = model(audio, motion, noise, genre)
    print(pred_motion.shape)
    # l_motion = l_motion[:, :, :-3]


    print("[*] Finish DanceGeneratorAMNG_D")
    print("*******************************")
