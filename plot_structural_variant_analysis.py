## SuPreMo-Akita_weighted_scores for plot

import os
import sys
import math
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from pybedtools import BedTool
from matplotlib.backends.backend_pdf import PdfPages

# === Setup paths and parameters ===

data_dir = '/home/reny2/beegfs/reny2/02work/p4_3Dgenome/Akita/01DS_GSE185192/ISM'
in_file = f'{data_dir}/ADRD_variants/Lee_SV.txt' # Input directory
#directory = f'{data_dir}Finetune_NPC_modelOut/Lee_ISMOut' # Output directory

## out for Mic_NPC model
directory = f'{data_dir}/Finetune_MiC_NPC_modelOut/Lee_ISMOut' # Output directory
figure_out = f'{data_dir}/Finetune_MiC_NPC_modelOut/Lee_ISMOut/figure'

file_prefix = 'Lee_weighted'
out_file = os.path.join(directory, file_prefix)

# Analysis configuration
shift = 0
revcomp_suffix = '_revcomp'
scoring_method = 'corr'
roi = 'genes'
roi_scales = 10
genome = 'hg38'


# === Load custom SuPreMo scripts ===
sys.path.insert(0, '/home/reny2/beegfs/reny2/02work/p4_3Dgenome/SuPreMo/scripts')
import reading_utils, plotting_utils, get_Akita_scores_utils, get_roi_utils

reading_utils.var_set_size = 10000000000
get_Akita_scores_utils.BedTool = BedTool
get_Akita_scores_utils.roi_coords_BED = get_roi_utils.get_roi(roi, genome)

# === Load data ===
print("📥 Loading input and outputs...")
variants = reading_utils.read_input(f'{in_file}', 0)
print("variants----",variants[:5])
scores = pd.read_csv(f'{out_file}_scores', sep = '\t')
scores = pd.concat([variants, scores], axis = 1)
#scores = pd.concat([variants, pd.read_csv(f'{out_file}_scores', sep = '\t')], axis = 1)

## Get map pairs where weights have strong effect on score
scores['weight_diff'] = [x-y for x,y in zip(scores['mse_10-weighted_HFF_0'], scores['mse_HFF_0'])]
print("Get map pairs where weights have strong effect on score------",scores.sort_values('weight_diff'))

# Sort by weight_diff in descending order (optional)
scores_sorted = scores.sort_values('weight_diff', ascending=False)
# Save as TSV (tab-separated file)
output_path = os.path.join(directory, "scores_with_weight_diff.tsv")
scores_sorted.to_csv(output_path, sep='\t', index=False)
print(f"✅ Saved scores with weight_diff to: {output_path}")



# Check and load map/track files
map_file = f'{out_file}_maps.npy'
track_file = f'{out_file}_tracks.npy'
if not os.path.exists(map_file) or not os.path.exists(track_file):
    raise FileNotFoundError("Required .npy files not found.")

maps = np.load(map_file, allow_pickle=True).item()
tracks = np.load(track_file, allow_pickle=True).item()

print(f"✅ Loaded {len(maps)} map entries and {len(tracks)} track entries")
print(f"✅ Number of unique variants: {scores['var_index'].nunique()}")
n = scores['var_index'].nunique()
print("variants number:",n)


# === Helper function to compute weight track ===
def get_weight_track(disruption_track, roi_in_map, roi_scale):
    # Get the bins that correspond regions of interest (roi)   
    roi_bins = []
    for i in range(len(roi_in_map)):      
        bins_i = list(range(roi_in_map.iloc[i].Start, roi_in_map.iloc[i].End + 1))
        roi_bins.append(bins_i)  
    roi_bins = np.unique([item for group in roi_bins for item in group if item < 448])   
    background_weight = 1
    # Get weight track  
    weight_track = np.array([background_weight]*len(disruption_track))
    for i in roi_bins:
        weight_track[i] = roi_scale
    # Add nans to weight track
    weight_track[np.isnan(disruption_track)] = 0
    return weight_track


# === Loop through variants ===

# disruption_pdf_path = os.path.join(figure_out, "all_variants_disruption_tracks.pdf")
# roi_pdf_path = os.path.join(figure_out, "all_variants_maps_roi.pdf")
mapGene_pdf_path = os.path.join(figure_out, "all_variants_mapGene_new.pdf")

# disruption_pdf = PdfPages(disruption_pdf_path)
# roi_pdf = PdfPages(roi_pdf_path)
mapGene_pdf = PdfPages(mapGene_pdf_path)

#with PdfPages(disruption_pdf_path) as disruption_pdf, PdfPages(roi_pdf_path) as roi_pdf, PdfPages(mapGene_pdf_path) as mapGene_pdf:
with PdfPages(mapGene_pdf_path) as mapGene_pdf:
    #for var_index in scores['var_index'].unique():
    for var_index in range(n):
        variant_id = f'{var_index}_maps_HFF_{shift}{revcomp_suffix}'
        track_id = f'{var_index}_{scoring_method}_track_HFF_{shift}{revcomp_suffix}'

        if variant_id not in maps or track_id not in tracks:
            print(f"⚠️  Skipping {var_index}:'{variant_id}' not found in maps or tracks.")
            continue
        
        # Load predictions
        REF_pred, ALT_pred = maps[variant_id][:2]
        REF_pred, ALT_pred = [get_Akita_scores_utils.map_from_vector(x) for x in [REF_pred, ALT_pred]]
        rel_pos_map = maps[variant_id][2]
        map_start_coord = maps[variant_id][3]

        # Extract variant metadata    
        row = scores[scores.var_index == var_index]
        CHR = row.CHROM.values[0]
        POS = row.POS.values[0]
        END = row.END.values[0] if 'END' in row.columns else POS + abs(row.SVLEN.values[0])
        SVTYPE = row.SVTYPE.values[0] if 'SVTYPE' in row.columns else '-'
        SVLEN = row.SVLEN.values[0] if 'SVLEN' in row.columns else '-'
        
        # Generate Title for Each Plot
        plot_title = f"{CHR}:{POS}-{END}, SV Type: {SVTYPE}, SV Length: {SVLEN}"
        disruption_track = tracks[track_id]

        # # === Plot disruption track ===
        # plt.figure(figsize=(10, 4))
        # plotting_utils.plot_disruption_tracks(disruption_track, scoring_method)
        # plt.title(plot_title)
        # #filename = os.path.join(figure_out, f"variant_{var_index}_disruption_track.pdf")
        # #plt.savefig(filename, format="pdf", bbox_inches="tight")
        # disruption_pdf.savefig(bbox_inches="tight")
        # plt.close('all')


        # === Plot map with ROI ===
        genes_in_map = plotting_utils.get_genes_in_map(CHR, map_start_coord, rel_pos_map, SVTYPE, SVLEN)
        if genes_in_map is None or genes_in_map.empty:
            print(f"⚠️ Skipping {var_index}: genes_in_map is empty or None.")
            continue
        genes_in_map = genes_in_map.reset_index(drop=True)
        # print(f"✅ genes_in_map shape: {genes_in_map.shape}")
        # print(f"✅ genes_in_map columns: {genes_in_map.columns}")
        # print(f"✅ genes_in_map index: {genes_in_map.index}")
        
        roi_in_map = get_Akita_scores_utils.get_roi_in_map(CHR, map_start_coord, rel_pos_map, SVTYPE, SVLEN)
        if not isinstance(roi_in_map, pd.DataFrame):
            print(f"❌ Invalid roi_in_map at var_index {var_index}: type={type(roi_in_map)}")
            continue
        weight_track = get_weight_track(disruption_track, roi_in_map, roi_scales)
        lines = plotting_utils.get_var_bins(rel_pos_map, SVTYPE, SVLEN)
        
        # ## plot roi map
        # plt.figure(figsize=(12, 8))
        # plotting_utils.plot_maps_roi([REF_pred, ALT_pred], genes_in_map, lines, roi_in_map, disruption_track, weight_track, scale=1, gene_rows=5)
        # plt.title(plot_title)
        # roi_pdf.savefig(bbox_inches="tight")
        # plt.close('all')

        # === Plot maps with genes and tracks ===
        plt.figure(figsize=(12, 6),constrained_layout=True,dpi=300)
        plotting_utils.plot_maps_genes_tracks_test([REF_pred, ALT_pred], genes_in_map, lines, disruption_track, scale=1, gene_rows=5)      
        plt.title(plot_title)
        mapGene_pdf.savefig(bbox_inches="tight")
        #filename = os.path.join(figure_out, f"variant_{var_index}_maps_genes_tracks.pdf")
        #plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close('all')

        # ### Plot maps with genes structure locus
        # plt.figure(figsize=(12, 6))
        # plotting_utils.plot_gene_locus(genes_in_map, scale=1, ax=None)
        # plt.title(plot_title)
        # filename = os.path.join(figure_out, f"GeneLocus_{var_index}.pdf")
        # plt.savefig(filename, format="pdf", bbox_inches="tight")
        # plt.close()


        # ### Plot maps with tracks
        # plt.figure(figsize=(10, 6))
        # plotting_utils.plot_maps_tracks([REF_pred, ALT_pred], lines, disruption_track, scale=1)
        # plt.title(plot_title)
        # pdf.savefig()
        # plt.close()

        ### Plot maps with genes
        #plt.figure(figsize=(10, 6))
        #plotting_utils.plot_maps_genes([REF_pred, ALT_pred], genes_in_map, lines, scale=1, gene_rows=5)
        #plt.title(plot_title)
        #filename = os.path.join(figure_out, f"variant_{var_index}_maps_genes.pdf")
        #plt.savefig(filename, format="pdf", bbox_inches="tight")
        #plt.close()

        ### Plot only maps
        #plt.figure(figsize=(10, 6))
        #plotting_utils.plot_maps([REF_pred, ALT_pred], lines, scale=1)
        #plt.title(plot_title)
        #filename = os.path.join(figure_out, f"variant_{var_index}_maps.pdf")
        #plt.savefig(filename, format="pdf", bbox_inches="tight")
        #plt.close()



