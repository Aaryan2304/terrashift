# Original OSCD data audit

This report was generated from the downloaded files by `scripts/audit_oscd_original.py`.
The script does not modify the source dataset.

## Verified structure

- Official split: 14 train regions and 10 test regions; lists are disjoint and cover all 24 regions.
- Scene folders: 24.
- Native image bands checked: 624; native CRS: EPSG:4326.
- Rectified image bands checked: 624; CRS is absent and transforms are identity pixel grids.
- Every rectified date pair has 13 named bands on one common shape. Each label has the same shape.
- Every label PNG agrees with the TIFF class map at a 128 intensity boundary, allowing antialiased visualization edges.

## Label encoding finding

The downloaded label TIFFs use values 1 (no change) and 2 (change) in all 24 regions. The label README says 0/1. The PNG visualization independently supports 1/2 mapping: class 1 is dark and class 2 is light. The loader must map class 2 to binary 1 and class 1 to binary 0. It must reject unknown values.

The project page describes the labels as pixel-level urban changes focused on new buildings and roads. It does not claim every positive pixel is all structural change types.

## Per-region counts

| Region | Split | First date | Second date | Shape (H×W) | No-change px | Change px | Encoding |
|---|---|---|---|---:|---:|---:|---|
| aguasclaras | train | date_1: 20150916 | date_2: 20171015 | 471×525 | 243219 | 4056 | observed_1_no_change_2_change |
| bercy | train | date_1: 20161130 | date_2: 20170829 | 395×360 | 141154 | 1046 | observed_1_no_change_2_change |
| bordeaux | train | date_1: 20160504 | date_2: 20171026 | 517×461 | 235957 | 2380 | observed_1_no_change_2_change |
| nantes | train | date_1: 20150821 | date_2: 20171014 | 522×582 | 300355 | 3449 | observed_1_no_change_2_change |
| paris | train | date_1: 20161130 | date_2: 20171107 | 408×390 | 158662 | 458 | observed_1_no_change_2_change |
| rennes | train | date_1: 20150821 | date_2: 20170621 | 339×563 | 185925 | 4932 | observed_1_no_change_2_change |
| saclay_e | train | date_1: 20160315 | date_2: 20170829 | 639×688 | 435286 | 4346 | observed_1_no_change_2_change |
| abudhabi | train | date_1: 20160120 | date_2: 20180328 | 799×785 | 603623 | 23592 | observed_1_no_change_2_change |
| cupertino | train | date_1: 20150918 | date_2: 20180326 | 1015×788 | 780878 | 18942 | observed_1_no_change_2_change |
| pisa | train | date_1: 20150704 | date_2: 20180211 | 776×718 | 548038 | 9130 | observed_1_no_change_2_change |
| beihai | train | date_1: 20161209 | date_2: 20180309 | 902×772 | 679026 | 17318 | observed_1_no_change_2_change |
| hongkong | train | date_1: 20160927 | date_2: 20180323 | 695×540 | 361921 | 13379 | observed_1_no_change_2_change |
| beirut | train | date_1: 20150820 | date_2: 20171003 | 1180×1070 | 1228682 | 33918 | observed_1_no_change_2_change |
| mumbai | train | date_1: 20151130 | date_2: 20180319 | 858×557 | 465662 | 12244 | observed_1_no_change_2_change |
| brasilia | test | date_1: 20150916 | date_2: 20171017 | 433×469 | 197835 | 5242 | observed_1_no_change_2_change |
| montpellier | test | date_1: 20150812 | date_2: 20171030 | 426×451 | 179072 | 13054 | observed_1_no_change_2_change |
| norcia | test | date_1: 20150711 | date_2: 20171018 | 241×385 | 91558 | 1227 | observed_1_no_change_2_change |
| rio | test | date_1: 20160424 | date_2: 20171011 | 353×426 | 141826 | 8552 | observed_1_no_change_2_change |
| saclay_w | test | date_1: 20160315 | date_2: 20170829 | 639×688 | 434614 | 5018 | observed_1_no_change_2_change |
| valencia | test | date_1: 20160730 | date_2: 20171107 | 458×476 | 217039 | 969 | observed_1_no_change_2_change |
| dubai | test | date_1: 20151211 | date_2: 20180330 | 774×634 | 442027 | 48689 | observed_1_no_change_2_change |
| lasvegas | test | date_1: 20150820 | date_2: 20180205 | 824×716 | 544714 | 45270 | observed_1_no_change_2_change |
| milano | test | date_1: 20161228 | date_2: 20180122 | 545×558 | 301691 | 2419 | observed_1_no_change_2_change |
| chongqing | test | date_1: 20170414 | date_2: 20180402 | 730×544 | 368483 | 28637 | observed_1_no_change_2_change |

## Sources

- OSCD authors’ project page: https://rcdaudt.github.io/oscd/
- Downloaded `Images/README.txt` and `Train Labels/README.txt` / `Test Labels/README.txt`.
- TorchGeo OSCD loader documentation: https://docs.torchgeo.org/en/stable/api/datasets/oscd.html

This audit verifies files and dimensions. It does not establish that a model performs well.
