# Dataset card: MILK10k clinical close-up subset

## Source

This project uses the clinical close-up half of the MILK10k training dataset from the International Skin Imaging Collaboration (ISIC) Archive. MILK10k contains paired clinical close-up and dermoscopic images for 5,240 lesions. Only the 5,240 clinical close-up images are used by this model because the intended demo accepts phone-like close-up photographs.

- Dataset DOI: https://doi.org/10.34970/648456
- Official description: https://api.isic-archive.com/doi/milk10k/
- License: Creative Commons Attribution-NonCommercial (CC-BY-NC)
- Required citation: “MILK study team. MILK10k. ISIC Archive, 2025, doi:10.34970/648456.”

## Label mapping

The model is a binary educational classifier, not a diagnostic system.

### Higher concern (`label = 1`)

- `AKIEC`: actinic keratosis / intraepidermal carcinoma
- `BCC`: basal cell carcinoma
- `MAL_OTH`: other malignant proliferations
- `MEL`: melanoma
- `SCCKA`: squamous cell carcinoma / keratoacanthoma

### Lower concern (`label = 0`)

- `BEN_OTH`: other benign proliferations
- `BKL`: benign keratinocytic lesions
- `DF`: dermatofibroma
- `INF`: inflammatory and infectious conditions
- `NV`: melanocytic nevus
- `VASC`: vascular lesions and hemorrhage

“Lower concern” does not mean cancer-free. It only means the image score fell below a validation threshold for this dataset and model.

## Splitting policy

- 80% training
- 10% validation
- 10% final test
- Stratified by binary label
- Grouped by lesion identifier so the same lesion cannot cross splits
- Fixed random seed for reproducibility

## Known limitations

- The dataset is enriched for suspicious lesions and does not represent cancer prevalence in the general population.
- Skin-tone groups are not evenly represented.
- A curated clinical close-up image can differ from a real user’s phone photograph.
- Device, lighting, focus, framing, and image-processing differences can change model performance.
- Several diagnosis categories have few examples.
- Dataset labels and a held-out test set do not constitute prospective clinical validation.

