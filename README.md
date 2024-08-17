# K-means filter for stylizing image textures
![example image](https://github.com/Alfred-N/Texture-Filters/blob/9c5e78784ca8d99f97de5b67e5be335b0c587953/example_comp.jpg)

## Install Conda env
`conda env create -f environment.yml`

`conda activate Filters`

## Example command

`python src/kmeans_filter.py -i "tree_bark.jpg" -o "out.jpg" --ref_path="flowers.jpg" --plot_comparison`

`python src/kmeans_filter.py -i T_Mossy_Forest_Boulder_wjwpfbf_1K_D.exr  -o "out.jpg" --plot_comparison --n_colors=4 --smoothing_type=gaussian --smoothing_strength=11 --post_process=median`