import os
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to checkpoint")
    parser.add_argument("--jz2cines", action="store_true")
    parser.add_argument("--remove-modality", choices=["audio", "text"], default="text")
    args = parser.parse_args()

    CINES_DATA_PATH = "/lus/home/CT10/c1615074/tphle/Data/prepared"
    JZ_DATA_PATH = "/gpfswork/rech/ahm/umz16dj/Data"
    if args.jz2cines:
        print(f'Converting data path from Jean Zay to Cines')
        OLD_DATA_PATH = JZ_DATA_PATH
        NEW_DATA_PATH = CINES_DATA_PATH
    else:
        OLD_DATA_PATH = CINES_DATA_PATH
        NEW_DATA_PATH = JZ_DATA_PATH
        print(f'Converting data path from Cines to Jean Zay')

    # change path of checkpoints
    ckpt = torch.load(args.path, map_location="cpu")
    # print(ckpt["cfg"])
    cfg = ckpt["cfg"]
    for k, v in cfg.items():
        # if isinstance(v, dict):
        #     for kk, vv in v.items():
        #         print(f"{k}: {kk}: {vv}")
        # else:
        #     print(f"{k}: {v}")
        if k == "task":
            for subtask, subtask_cfg in v.items():
                if subtask != args.remove_modality:
                    if isinstance(subtask_cfg, dict):
                        cfg[k][subtask]["data"] = subtask_cfg["data"].replace(OLD_DATA_PATH, NEW_DATA_PATH)
                        print(f"{subtask}: {subtask_cfg}")
                else:
                    cfg[k][subtask] = None
                    print(f"{subtask}: {cfg[k][subtask]}")
    ckpt["cfg"] = cfg
    saved_path = f'{args.path.replace(".pt", "_updated_path.pt")}'
    torch.save(ckpt, saved_path)
    print(f"saved updated checkpoint to {saved_path}")

if __name__ == "__main__":
    main()