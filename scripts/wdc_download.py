import pandas as pd
from datasets import load_dataset

def export_wdc_cameras(output_dir: str = "../datasets/wdc", name: str = "watches_small"):
    output_dir = output_dir + f"/{name}"
    lsource_data = []
    rsource_data = []
    for split in ["train", "test"]:
        # Carica dataset da Hugging Face
        ds = load_dataset("wdc/products-2017", name, split=split)
        # Converti in DataFrame pandas
        df = pd.DataFrame(ds)

        # Le colonne seguono naming: id_left, id_right, label, title_left, title_right, etc.
        # Costruisci tableA e tableB: unique offerte left/right
        left_cols = [c for c in df.columns if c.endswith("_left")]
        right_cols = [c for c in df.columns if c.endswith("_right")]

        df_left = df[left_cols].copy()
        df_right = df[right_cols].copy()

        # Normalizza nomi colonne (rimuovi _left / _right)
        df_left.columns = [c[:-5] for c in df_left.columns]
        df_right.columns = [c[:-6] for c in df_right.columns]  # "_right" è 6 caratteri

        # Aggiungi una colonna “offer_id” per ogni tabella
        df_left = df_left.rename(columns={"id": "offer_id"})
        df_right = df_right.rename(columns={"id": "offer_id"})

        # Salva tableA e tableB con gli offer_id
        import os
        os.makedirs(output_dir, exist_ok=True)
        tableA_path = os.path.join(output_dir, f"tableA_{split}.csv")
        tableB_path = os.path.join(output_dir, f"tableB_{split}.csv")
        df_left.to_csv(tableA_path, index=False)
        df_right.to_csv(tableB_path, index=False)
        lsource_data.append(df_left)
        rsource_data.append(df_right)

        # Crea train.csv con ltable_id, rtable_id, label
        train = df[["id_left", "id_right", "label"]].copy()
        train = train.rename(columns={"id_left": "ltable_id", "id_right": "rtable_id"})
        train_path = os.path.join(output_dir, f"{split}.csv")
        train.to_csv(train_path, index=False)

        print("Saved:")
        print("  ", tableA_path)
        print("  ", tableB_path)
        print("  ", train_path)
    pd.concat(lsource_data).to_csv(os.path.join(output_dir, "tableA.csv"), index=False)
    pd.concat(rsource_data).to_csv(os.path.join(output_dir, "tableB.csv"), index=False)

if __name__ == "__main__":
    export_wdc_cameras()
