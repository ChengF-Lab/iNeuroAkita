#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run script for ChIP-seq and Contact Map Plotting

This script demonstrates how to use the plotting functions from Plot_ChIP_Map.py
"""

import os
from Plot_ChIP_Map_Improved_v1 import (
    PlotConfig,
    load_data,
    plot_one_region,
    plot_multi_region,
    quick_plot,
    load_config_from_json,
    save_config_to_json
)


# =====================
# Method 1: Basic Usage with PlotConfig
# =====================
def example_basic_usage():
    """Basic usage example with PlotConfig."""
    
    # Step 1: Create configuration
    config = PlotConfig(
        # Required: File paths
        ctrl_npz_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/predictOut_NPC_P16_CTCF_fasta/akita_fasta_predictions_targets.npz",
        mut_npz_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/predictOut_NPC_P16_mutated_CTCF_fasta/mutated_predictions_avg.npz",
        bw_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/CTCF_Mutated_disruption/NPC_CTCF_ChIP/ENCFF782OKU.bigWig",
        output_dir="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/CTCF_Mutated_disruption/JASPAR_out",
        # Required: Data parameters
        target_index=0,
        target_name="NPC_control_male",
        
        # Optional: Title spacing (bring title closer to plots)
        suptitle_y=0.975,  # Title vertical position (0.95-0.99, lower = closer)
        top_margin=0.96,   # Top margin (0.94-0.98, lower = less space)
        save_pad_inches=0.05,  # Padding when saving (0.02-0.1, lower = tighter)
        
        # Optional: Customize appearance
        # vmin_main=-2.0, vmax_main=2.0,  # Color scale
        # sigma=2.0,  # Smoothing
        # diff_mode="pred",  # "pred" or "target"
        # show_metrics=True,  # Show correlation metrics
        # gene_band_alpha=0.25,  # Gene annotation transparency
    )
    
    # Step 2: Load data (only once, can be reused for multiple regions)
    print("Loading data...")
    region_labels, targets, preds_ctrl, preds_mut, bw = load_data(config)
    print(f"✅ Loaded {len(region_labels)} regions")
    
    # Step 3: Define gene marks (optional)
    gene_marks = [
        {"name": "FABP5", "start": 81280536, "end": 81284775, "color": "#e6b800"},
        {"name": "IMPA1", "start": 81656914, "end": 81686325, "color": "#e6b800"},
        {"name": "SNX16", "start": 81799583, "end": 81842185, "color": "#e6b800"},
        # Add more genes as needed:
        # {"name": "Gene2", "start": 81300000, "end": 81310000, "color": "#ff0000"},
    ]
    
    # Step 4: Plot single region
    region = "chr8:80656384-81704960"
    output_path = plot_one_region(
        region=region,
        config=config,
        region_labels=region_labels,
        targets=targets,
        preds_ctrl=preds_ctrl,
        preds_mut=preds_mut,
        bw=bw,
        gene_marks=gene_marks,
        output_path=os.path.join(config.output_dir, "Final_region_chr8_ChIP_ContactMap_WITH_MUT_DIFF_metrics_CBbottom_251205.pdf")
    )
    print(f"✅ Single region PDF saved to: {output_path}")
    
    # Step 5: Plot multiple regions

    ## top10  by Disruption_Score_Mean
    regions = [
        "chr8:80656384-81704960",
        "chr5:76333056-77381632",
        "chr1:30457856-31506432",
        "chr5:73187328-74235904",
        "chr20:8429568-9478144",
        "chr8:80132096-81180672",
        "chr9:32282624-33331200",
        "chr17:49166336-50214912",
        "chr6:141787136-142835712",
        "chr16:11603968-12652544"
    ]



    ## top10 Delta_Signal_Mean
    #regions = [
    #   "chr20:8429568-9478144",
    #   "chr18:61196288-62244864",
    #   "chr20:7905280-8953856",
    #   "chr5:73187328-74235904",
    #   "chr2:135497728-136546304",
    #   "chr10:85647360-86695936",
    #   "chr2:44967936-46016512",
    #   "chr2:47065088-48113664",
    #   "chr11:78051328-79099904",
    #   "chr17:57030656-58079232"
    #]

    
    gene_marks_by_region = {
        "chr8:80656384-81704960": gene_marks,
        "chr5:76333056-77381632":[
            {"name": "SV2C", "start": 76083383, "end": 76333956, "color": "#e6b800"},
            {"name": "S100Z", "start": 76850014, "end": 76921650, "color": "#e6b800"}
            ],
        "chr1:30457856-31506432":[
            {"name": "SDC3", "start": 30869466, "end": 30908758, "color": "#e6b800"},
            {"name": "NKAIN1", "start": 31179745, "end":31239887, "color": "#e6b800"},
            ],
        "chr5:73187328-74235904":[
            {"name": "FOXD1", "start": 73446266, "end": 73448777, "color": "#e6b800"},
            {"name": "ARHGEF28", "start": 73626196, "end": 73941990, "color": "#e6b800"},
            ],
        "chr20:8429568-9478144":[
            {"name": "PLCB1", "start": 8132266, "end": 8884900, "color": "#e6b800"}
            ],


        # Add gene marks for other regions if needed
        # "chr5:76333056-77381632": [{"name": "GeneX", "start": ..., "end": ..., "color": "..."}],
    }
    
    output_path = plot_multi_region(
        regions=regions,
        config=config,
        region_labels=region_labels,
        targets=targets,
        preds_ctrl=preds_ctrl,
        preds_mut=preds_mut,
        bw=bw,
        gene_marks_by_region=gene_marks_by_region,
        output_path=os.path.join(config.output_dir, "Final_MultiRegion_list1_WITH_MUT_DIFF_metrics_CBbottom_251205.pdf")
    )
    print(f"✅ Multi-region PDF saved to: {output_path}")


# =====================
# Method 2: Quick Plot (Simplified)
# =====================
def example_quick_plot():
    """Quick plot example with minimal parameters."""
    
    # All in one function call
    output_path = quick_plot(
        region="chr8:80656384-81704960",
        ctrl_npz_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/predictOut_NPC_P16_CTCF_fasta/akita_fasta_predictions_targets.npz",
        mut_npz_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/predictOut_NPC_P16_mutated_CTCF_fasta/mutated_predictions_avg.npz",
        bw_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/CTCF_Mutated_disruption/NPC_CTCF_ChIP/ENCFF782OKU.bigWig",
        output_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/CTCF_Mutated_disruption/Plot/quick_plot_output.pdf",
        target_index=0,
        target_name="NPC_control_male",
        gene_marks=[
            {"name": "FABP5", "start": 81280536, "end": 81284775, "color": "#e6b800"},
        ],
        # Optional: Override config parameters
        suptitle_y=0.975,  # Title closer to plots
        top_margin=0.96,   # Less top space
    )
    print(f"✅ Quick plot saved to: {output_path}")


# =====================
# Method 3: Using JSON Config File
# =====================
def example_json_config():
    """Example using JSON configuration file."""
    
    # Step 1: Create and save config to JSON
    config = PlotConfig(
        ctrl_npz_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/predictOut_NPC_P16_CTCF_fasta/akita_fasta_predictions_targets.npz",
        mut_npz_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/predictOut_NPC_P16_mutated_CTCF_fasta/mutated_predictions_avg.npz",
        bw_path="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/CTCF_Mutated_disruption/NPC_CTCF_ChIP/ENCFF782OKU.bigWig",
        output_dir="/home/reny2/isilon/Cheng-Ren/pOther/bulkHiC/akita_pipeline/03model_input/CTCF_Mutated_disruption/Plot",
        target_index=0,
        target_name="NPC_control_male",
        suptitle_y=0.975,
        top_margin=0.96,
    )
    
    # Save config to JSON file
    config_json_path = "plot_config.json"
    save_config_to_json(config, config_json_path)
    print(f"✅ Config saved to: {config_json_path}")
    
    # Step 2: Load config from JSON (for future use)
    config = load_config_from_json(config_json_path)
    print(f"✅ Config loaded from: {config_json_path}")
    
    # Step 3: Use config as before
    region_labels, targets, preds_ctrl, preds_mut, bw = load_data(config)
    
    # Plot
    output_path = plot_one_region(
        region="chr8:80656384-81704960",
        config=config,
        region_labels=region_labels,
        targets=targets,
        preds_ctrl=preds_ctrl,
        preds_mut=preds_mut,
        bw=bw,
        gene_marks=[
            {"name": "FABP5", "start": 81280536, "end": 81284775, "color": "#e6b800"},
        ],
    )
    print(f"✅ PDF saved to: {output_path}")


# =====================
# Method 4: Custom Configuration
# =====================
def example_custom_config():
    """Example with custom configuration parameters."""
    
    # Create config with custom parameters
    config = PlotConfig(
        # File paths
        ctrl_npz_path="/path/to/control.npz",
        mut_npz_path="/path/to/mutated.npz",
        bw_path="/path/to/chip.bigWig",
        output_dir="/path/to/output",
        
        # Data parameters
        target_index=0,
        target_name="My_Cell_Type",
        
        # Custom appearance
        figsize=(8, 24),  # Larger figure
        vmin_main=-3.0,   # Wider color scale
        vmax_main=3.0,
        vmin_diff=-2.0,
        vmax_diff=2.0,
        sigma=3.0,        # More smoothing
        
        # Title spacing
        suptitle_y=0.97,  # Title closer
        top_margin=0.95,  # Less top space
        save_pad_inches=0.03,  # Minimal padding
        
        # Gene annotations
        gene_band_alpha=0.3,      # More visible
        gene_label_offset=0.06,   # Further from plot
        gene_label_fontsize=8,    # Larger labels
        
        # Show/hide panels
        show_target=True,
        show_predicted=True,
        show_mutated=True,
        show_diff=True,
        show_chip=True,
        show_colorbars=True,
        show_metrics=True,
        show_mse=True,
    )
    
    # Use config as before
    region_labels, targets, preds_ctrl, preds_mut, bw = load_data(config)
    # ... plot as usual


# =====================
# Main Execution
# =====================
if __name__ == "__main__":
    # Choose which example to run:
    
    # Method 1: Basic usage (recommended)
    example_basic_usage()
    
    # Method 2: Quick plot (simplest)
    # example_quick_plot()
    
    # Method 3: JSON config (for reusable configs)
    # example_json_config()
    
    # Method 4: Custom config (for advanced usage)
    # example_custom_config()

