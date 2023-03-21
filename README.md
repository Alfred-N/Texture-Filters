# K-means filter for stylizing image textures
![example image](https://github.com/Alfred-N/Texture-Filters/blob/9c5e78784ca8d99f97de5b67e5be335b0c587953/example_comp.jpg)

## Install Conda env
`conda env create -f environment.yml`
`conda activate Filters`

## Example command

`python kmeans_filter.py -i "tree_bark.jpg" -o "out.jpg" --ref_path="flowers.jpg" --plot_comparison`
