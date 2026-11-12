#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_framework.py
=================
One-Click Generator: Jupyter Notebook (.ipynb) -> Standalone CLI Script (.py)

Automatically converts a notebook in two phases:
  Phase 1 - BUILD:   Patch of the critical cells (CLI argparse, headless matplotlib,
                     output dir, fix of hardcoded paths, global plt.show).
  Phase 2 - CONVERT: Conversion .ipynb -> .py with sanitization of the Jupyter commands.

USAGE (from the project root):
    python sync_framework.py
"""

# ==============================================================================
# CONFIGURATION  <--- EDIT ONLY THIS SECTION
# ==============================================================================

# Name of the source notebook (relative to the project root)
SOURCE_NOTEBOOK = "InVisible.ipynb"

# Destination folder for the generated files
OUTPUT_DIR_FRAMEWORK = "standalone_framework"

# Base name (without extension) for the generated files
OUTPUT_BASENAME = "InVisible"

# Indices of the cells that require specific patches.
# Static fallback values — overwritten by the auto-detect at runtime (see _autodetect_cell_indices).
CELL_INDEX_SETUP        = 1   # Environment configuration (pip install, matplotlib)
CELL_INDEX_MASTER_CFG   = 3   # Master Configuration Cell (GLOBAL_CONFIG, EXECUTION_MODE)
CELL_INDEX_MODEL_ARCH   = 18  # MODEL_CONFIG architecture  (auto-detected from 'MODEL_ARCH')
CELL_INDEX_INFERENCE    = 28  # INFERENCE_EVAL             (auto-detected from 'INFERENCE_EVAL')
CELL_INDEX_EXPORT_CSV   = 98  # Final CSV export          (auto-detected from 'RUN_RESULTS'/'benchmark')

# ==============================================================================
# END OF CONFIGURATION
# ==============================================================================

import sys
import json
import os
import datetime


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# argparse CLI block to inject at the start of the master config cell
# ---------------------------------------------------------------------------
_CLI_HEADER = '''\
# ==============================================================================
# CLI ARGUMENT PARSER & CENTRALIZED CONFIGURATION OF THE 5 PARAMETERS
# (Automatically injected by sync_framework.py -- do not edit by hand)
# ==============================================================================
import os
import sys
import argparse
import datetime
import logging

def parse_cli_args():
    """5 parameters from the terminal: dataset_path, model_path, mode, arch, training_resolution."""
    is_ipykernel = any("ipykernel" in str(a).lower() or "-f" in str(a) for a in sys.argv)
    parser = argparse.ArgumentParser(
        description="Unified Benchmark and XAI Forensic Diagnostics Framework for Backdoors"
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="blind_audit",
                        choices=["blind_audit", "only_dataset", "only_model", "dataset_only",
                                 "model_only", "attacker_model_level", "attacker_image_level"])
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--training_resolution", type=str, default="224x224",
                        choices=["32x32", "224x224"])
    if not is_ipykernel and len(sys.argv) > 1:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args([])
    return args

CLI_ARGS = parse_cli_args()

_mode_raw = str(CLI_ARGS.mode).lower()
if _mode_raw in ["blind_audit"]:
    EXECUTION_MODE = "BLIND_AUDIT"
elif _mode_raw in ["only_dataset", "dataset_only"]:
    EXECUTION_MODE = "DATASET_ONLY"
elif _mode_raw in ["only_model", "model_only"]:
    EXECUTION_MODE = "MODEL_ONLY"
elif _mode_raw in ["attacker_model_level"]:
    EXECUTION_MODE = "ATTACKER_MODEL_LEVEL"
elif _mode_raw in ["attacker_image_level"]:
    EXECUTION_MODE = "ATTACKER_IMAGE_LEVEL"
else:
    EXECUTION_MODE = "BLIND_AUDIT"

SELECTED_ARCH = str(CLI_ARGS.arch).lower()
SELECTED_RES_STR = str(CLI_ARGS.training_resolution).lower()
if SELECTED_RES_STR == "32x32":
    SELECTED_RES_TUPLE = (32, 32)
    SELECTED_STRATEGY = "scratch"
else:
    SELECTED_RES_TUPLE = (224, 224)
    SELECTED_STRATEGY = "pipeline"

SELECTED_DATASET_PATH = CLI_ARGS.dataset_path if CLI_ARGS.dataset_path else globals().get("PATH_DATASET", "CIFAR10")
SELECTED_MODEL_PATH = CLI_ARGS.model_path if CLI_ARGS.model_path else globals().get("IMPORTED_MODEL_PATH", "")

_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(os.getcwd(), f"output_{EXECUTION_MODE.lower()}_{_run_timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_log_file = os.path.join(OUTPUT_DIR, "run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Framework")
logger.info(f"=== RUN START: {EXECUTION_MODE} ===")
logger.info(f"Output Directory: {OUTPUT_DIR}")
logger.info(f"Dataset Path: {SELECTED_DATASET_PATH}")
logger.info(f"Model Path: {SELECTED_MODEL_PATH}")
logger.info(f"Architecture: {SELECTED_ARCH}")
logger.info(f"Training Resolution: {SELECTED_RES_STR} ({SELECTED_STRATEGY})")

'''

# CLI override block after GLOBAL_CONFIG (injected by sync_framework after the cell body)
_CLI_POST_OVERRIDE = '''
# ==============================================================================
# CLI OVERRIDE POST-GLOBAL_CONFIG
# Overrides the hardcoded values of GLOBAL_CONFIG with those passed from the terminal.
# MUST be executed AFTER the body of the master config cell.
# ==============================================================================
if CLI_ARGS.model_path:
    IMPORTED_MODEL_PATH = CLI_ARGS.model_path
    USE_IMPORTED_MODEL  = True
    if isinstance(globals().get("GLOBAL_CONFIG"), dict):
        GLOBAL_CONFIG["imported_model_path"] = IMPORTED_MODEL_PATH
    if isinstance(globals().get("PATH_CONFIG"), dict):
        PATH_CONFIG["best_model_path"] = IMPORTED_MODEL_PATH
    logger.info(f"[CLI OVERRIDE] IMPORTED_MODEL_PATH -> {IMPORTED_MODEL_PATH}")

if CLI_ARGS.dataset_path:
    PATH_DATASET = CLI_ARGS.dataset_path
    PATH_KAGGLE  = PATH_DATASET
    if isinstance(globals().get("GLOBAL_CONFIG"), dict):
        GLOBAL_CONFIG["dataset_path"] = PATH_DATASET
    if isinstance(globals().get("DATASET_CONFIG"), dict):
        DATASET_CONFIG["dataset_path"] = PATH_DATASET
    logger.info(f"[CLI OVERRIDE] PATH_DATASET -> {PATH_DATASET}")

# Re-apply OUTPUT_DIR in GLOBAL_CONFIG if present
if isinstance(globals().get("GLOBAL_CONFIG"), dict):
    GLOBAL_CONFIG["output_dir"] = OUTPUT_DIR
if isinstance(globals().get("PATH_CONFIG"), dict):
    PATH_CONFIG["output_dir"] = OUTPUT_DIR
'''

# Map internal section -> descriptive file name (used by _build_section_names_map)
_SECTION_NAME_MAP = {
    'DATASET_LOADER':          'dataset_samples',
    'PERCEPTUAL_METRICS':     'perceptual_metrics',
    'LPIPS_METRICS':           'lpips_perceptual_diff',
    'MODEL_ARCH':              'model_architecture',
    'TRAIN':                   'training_loss_curves',
    'INFERENCE_EVAL':          'confusion_matrix_eval',
    'GRADCAM':                 'gradcam_comparative',
    'NUMERIC_METRICS':      'numerical_metrics',
    'CONFIDENCE_AIP':          'confidence_aip',
    'CONFIDENCE':              'confidence_distributions',
    'LATENCY':                 'latency_benchmarks',
    'POISONING':               'poisoning_latency_profile',
    'CAPTUM_SUITE':            'captum_suite',
    'TCAV':                    'tcav_concept_report',
    'CAPTUM_ROBUST':           'captum_robust_adversarial',
    'TRACIN':                  'tracin_influence_samples',
    'RISE':                    'rise_saliency_panel',
    'ACTIVATION_MAXIMIZATION': 'am_prototypes_all_classes',
    'TRIGGER_AWARE_AM':        'trigger_aware_am_forensics',
    'ZHU_CANGELOSI':           'zhu_cangelosi_am',
    'ZHU_CANGELOSI_PIP':       'zhu_cangelosi_pip',
    'SOLEFLIP':                'zhu_cangelosi_am',
    'HYBRID_AM':               'hybrid_guided_am',
    'UCLC_LIPSCHITZ':          'uclc_lipschitz_dist',
    'TAC':                     'tac_change_dist',
    'CROSS_UCLC_TAC':          'cross_uclc_tac_val',
    'FCT':                     'fct_consistency_dist',
    'ACTIVATION_CLUSTERING':   'activation_clustering_pca',
    'SPECTRAL_SIGNATURES':     'spectral_signatures_svd',
    'CROSS_LATENT':            'cross_latent_triangulation',
    'FEDNIA':                  'fednia_soleflip_diag',
    'NEURAL_CLEANSE':          'neural_cleanse_mad',
    'ABS':                     'abs_stimulation_analysis',
    'SENTINET':                'sentinet_patch_forensics',
    'BYE':                     'bye_gallery_and_response',
    'NEURAL_COLLAPSE':         'neural_collapse_dashboard',
    'TABOR':                   'tabor_optimization_results',
}


def _build_section_names_map(nb):
    """
    Auto-detects the {cell_idx: descriptive_name} map by scanning the
    should_run_section('SECTION') calls in the source notebook.
    Guarantees correct names regardless of the notebook version.
    """
    import re
    result = {}
    seen_idx = set()
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        sections = re.findall(r"should_run_section\('([^']+)'\)", src)
        for sec in sections:
            if i not in seen_idx:
                name = _SECTION_NAME_MAP.get(sec, sec.lower())
                result[i] = name
                seen_idx.add(i)
                break  # uses only the first section of the cell
    return result


def _autodetect_cell_indices(nb):
    """
    Auto-detects the indices of the critical cells for the patches.
    Returns a dict with keys: setup, master_cfg, model_arch, inference, export_csv.
    Each rule finds the FIRST code cell that satisfies all the criteria.
    """
    indices = {
        'setup':      CELL_INDEX_SETUP,
        'master_cfg': CELL_INDEX_MASTER_CFG,
        'model_arch': CELL_INDEX_MODEL_ARCH,
        'inference':  CELL_INDEX_INFERENCE,
        'export_csv': CELL_INDEX_EXPORT_CSV,
    }
    detect_rules = {
        # Setup cell: pip install or matplotlib, does NOT assign GLOBAL_CONFIG nor contain should_run_section
        'setup': lambda s: (
            ('pip install' in s or ('matplotlib' in s and 'import' in s)) and
            'GLOBAL_CONFIG = {' not in s and 'GLOBAL_CONFIG={' not in s and
            'should_run_section' not in s
        ),
        # Master config cell: assigns GLOBAL_CONFIG as a literal dict (GLOBAL_CONFIG = {)
        # This is more precise than searching only for 'GLOBAL_CONFIG', which is everywhere
        'master_cfg': lambda s: (
            ('GLOBAL_CONFIG = {' in s or 'GLOBAL_CONFIG={' in s) and
            'EXECUTION_MODE' in s
        ),
        # MODEL_ARCH: first cell with should_run_section MODEL_ARCH
        'model_arch': lambda s: "should_run_section('MODEL_ARCH')" in s,
        # INFERENCE_EVAL: should_run_section + clean_memory in the same cell
        'inference':  lambda s: (
            "should_run_section('INFERENCE_EVAL')" in s and
            'clean_memory' in s
        ),
        # Final CSV export: to_csv + RUN_RESULTS, NOT a should_run_section section
        'export_csv': lambda s: (
            ('to_csv' in s or '.csv' in s.lower()) and
            'RUN_RESULTS' in s and
            'should_run_section' not in s
        ),
    }
    for key, rule in detect_rules.items():
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            cell_src = ''.join(cell['source'])
            try:
                if rule(cell_src):
                    indices[key] = i
                    break
            except Exception:
                pass
    return indices


# Global map (will be overwritten by _build_section_names_map at build time)
CELL_SECTION_NAMES = {}


def _plt_show_replacement(indent, cell_idx, counter):
    """
    Generates the plt.show() replacement block for cells that do not already save figures.
    Directs output to NOTEBOOK_IMAGES_DIR (or MODULES_OUTPUTS_DIR for modules) instead of polluting root OUTPUT_DIR.
    """
    pad = " " * indent
    sec_name = CELL_SECTION_NAMES.get(cell_idx, f"cell_{cell_idx:02d}")
    return (
        f"{pad}_dest_dir_{cell_idx}_{counter} = globals().get('NOTEBOOK_IMAGES_DIR', globals().get('OUTPUT_DIR', '.'))\n"
        f"{pad}os.makedirs(_dest_dir_{cell_idx}_{counter}, exist_ok=True)\n"
        f"{pad}_fig_path_{cell_idx}_{counter} = os.path.join(\n"
        f"{pad}    _dest_dir_{cell_idx}_{counter}, \n"
        f"{pad}    f'{sec_name}_{counter:02d}.png'\n"
        f"{pad})\n"
        f"{pad}plt.savefig(_fig_path_{cell_idx}_{counter}, bbox_inches='tight', dpi=150)\n"
        f"{pad}plt.close()"
    )


def _cm_save_replacement(indent):
    """Generates the confusion matrix saving block with the correct indentation inside NOTEBOOK_IMAGES_DIR."""
    pad = " " * indent
    return (
        f"\n{pad}# Automatic saving of the confusion matrix\n"
        f"{pad}_cm_dir = globals().get('NOTEBOOK_IMAGES_DIR', globals().get('OUTPUT_DIR', '.'))\n"
        f"{pad}os.makedirs(_cm_dir, exist_ok=True)\n"
        f"{pad}cm_save_path = os.path.join(_cm_dir, \"confusion_matrix.png\")\n"
        f"{pad}fig.savefig(cm_save_path, bbox_inches=\"tight\", dpi=200)\n"
        f"{pad}plt.close(fig)\n"
        f"{pad}print(f\"Confusion matrix saved to: {{cm_save_path}}\")\n"
    )


# ==============================================================================
# PHASE 1 - PATCH of the critical cells
# ==============================================================================

def _patch_cell_setup(nb, idx):
    """Setup cell: headless header + comments out the Jupyter magic commands (!, %)."""
    src = "".join(nb["cells"][idx]["source"])
    adapted = []
    for line in src.splitlines(keepends=True):
        s = line.strip()
        if s.startswith("!") or s.startswith("%"):
            adapted.append("# [HEADLESS/CLI ADAPTED] " + line)
        else:
            adapted.append(line)
    header = (
        "# ============================================================\n"
        "# 0. HEADLESS MATPLOTLIB CONFIGURATION & PERSISTENT LOGGING\n"
        "# ============================================================\n"
        "import os\n"
        "import sys\n"
        "import logging\n"
        "import matplotlib\n"
        "try:\n"
        "    matplotlib.use(\"Agg\")\n"
        "except Exception:\n"
        "    pass\n\n"
    )
    nb["cells"][idx]["source"] = [header + "".join(adapted)]
    log(f"  OK  Cell {idx} (Setup): headless header + magics commented out.")


def _patch_cell_master_config(nb, idx):
    """Master config cell: injects the CLI header, dynamically replaces Kaggle paths and configs."""
    import re
    src = "".join(nb["cells"][idx]["source"])

    # 1. EXECUTION_MODE -> commented out so CLI_ARGS sets it before GLOBAL_CONFIG
    src, n_exe = re.subn(
        r"^(\s*EXECUTION_MODE\s*=\s*['\"][^'\"]+['\"])",
        r"# \1  # Replaced by CLI_ARGS (see block above)",
        src,
        flags=re.MULTILINE
    )
    if n_exe > 0:
        log("      -> Replaced: EXECUTION_MODE")
    else:
        log("      ~  Skipped (not found): EXECUTION_MODE")

    # 2. Dynamic replacement of imported_model_path in GLOBAL_CONFIG
    src, n_mdl = re.subn(
        r'("imported_model_path"\s*:\s*)(?:r?["\'][^"\']*["\'])',
        r'\1SELECTED_MODEL_PATH',
        src
    )
    if n_mdl > 0:
        log("      -> Replaced: imported_model_path (dynamic regex)")
    else:
        log("      ~  Skipped (not found): imported_model_path")

    # 3. Dynamic replacement of dataset_path in GLOBAL_CONFIG
    src, n_ds = re.subn(
        r'("dataset_path"\s*:\s*)(?:r?["\'][^"\']*["\'])',
        r'\1SELECTED_DATASET_PATH',
        src
    )
    if n_ds > 0:
        log("      -> Replaced: dataset_path (dynamic regex)")
    else:
        log("      ~  Skipped (not found): dataset_path")

    # 4. Dynamic replacement of output_dir in GLOBAL_CONFIG
    src, n_out = re.subn(
        r'("output_dir"\s*:\s*)(?:r?["\'][^"\']*["\'])',
        r'\1OUTPUT_DIR',
        src
    )
    if n_out > 0:
        log("      -> Replaced: output_dir (dynamic regex)")
    else:
        log("      ~  Skipped (not found): output_dir")

    # 5. Dynamic replacement of architecture in GLOBAL_CONFIG
    src, n_arc = re.subn(
        r'("architecture"\s*:\s*)(?:r?["\'][^"\']*["\'])',
        r'\1SELECTED_ARCH',
        src
    )
    if n_arc > 0:
        log("      -> Replaced: architecture (dynamic regex)")
    else:
        log("      ~  Skipped (not found): architecture")

    nb["cells"][idx]["source"] = [_CLI_HEADER + "\n" + src + "\n" + _CLI_POST_OVERRIDE]
    log(f"  OK  Cell {idx} (Master Config): CLI header + paths replaced + post-override injected.")


def _patch_cell_model_arch(nb, idx):
    """MODEL_ARCH cell: uses SELECTED_ARCH for the architecture."""
    src = "".join(nb["cells"][idx]["source"])

    old = '"architecture": GLOBAL_CONFIG.get("architecture", "resnet18") if "GLOBAL_CONFIG" in globals() else "resnet18",'
    new = '"architecture": globals().get("SELECTED_ARCH", GLOBAL_CONFIG.get("architecture", "resnet18") if "GLOBAL_CONFIG" in globals() else "resnet18"),'

    if old in src:
        src = src.replace(old, new)
        log(f"  OK  Cell {idx} (Model Arch): architecture uses SELECTED_ARCH.")
    else:
        # Fallback version without the globals guard
        old2 = '"architecture": GLOBAL_CONFIG.get("architecture", "resnet18")'
        new2 = '"architecture": globals().get("SELECTED_ARCH", GLOBAL_CONFIG.get("architecture", "resnet18"))'
        if old2 in src:
            src = src.replace(old2, new2)
            log(f"  OK  Cell {idx} (Model Arch) [fallback]: architecture uses SELECTED_ARCH.")
        else:
            log(f"  ~   Cell {idx} (Model Arch): target not found, skip.")

    nb["cells"][idx]["source"] = [src]


def _patch_cell_inference(nb, idx):
    """INFERENCE_EVAL cell: arch_name uses SELECTED_ARCH; plt.show() -> save+close."""
    src = "".join(nb["cells"][idx]["source"])

    old_arch = "arch_name = GLOBAL_CONFIG.get('architecture', 'resnet50') if 'GLOBAL_CONFIG' in globals() else 'resnet50'"
    new_arch = "arch_name = globals().get('SELECTED_ARCH', GLOBAL_CONFIG.get('architecture', 'resnet18') if 'GLOBAL_CONFIG' in globals() else 'resnet18')"
    if old_arch in src:
        src = src.replace(old_arch, new_arch)
        log(f"  OK  Cell {idx} (Inference): arch_name uses SELECTED_ARCH.")
    else:
        log(f"  ~   Cell {idx} (Inference): arch_name target not found.")

    if "plt.show()" in src:
        lines_out = []
        for l in src.splitlines(keepends=True):
            if l.rstrip().endswith("plt.show()"):
                indent = len(l) - len(l.lstrip())
                replacement = _cm_save_replacement(indent)
                lines_out.append(replacement)
            else:
                lines_out.append(l)
        src = "".join(lines_out)
        log(f"  OK  Cell {idx} (Inference): plt.show() -> save+close.")

    nb["cells"][idx]["source"] = [src]


def _patch_cell_export_csv(nb, idx):
    """Export CSV cell: removes the ATTACKER_MODEL_LEVEL guard, adapts the file name."""
    src = "".join(nb["cells"][idx]["source"])

    old_guard = "if current_mode != 'ATTACKER_MODEL_LEVEL':"
    new_guard = (
        "# Universal export for all modes (BLIND_AUDIT, DATASET_ONLY, MODEL_ONLY)\n"
        "if False:  # guard removed by sync_framework.py"
    )
    if old_guard in src:
        src = src.replace(old_guard, new_guard)
        log(f"  OK  Cell {idx} (Export CSV): ATTACKER_MODEL_LEVEL guard removed.")
    else:
        log(f"  ~   Cell {idx} (Export CSV): guard not found.")

    old_fname = "f'run_{current_mode.lower()}.csv'"
    new_fname = "'run_blind_audit.csv' if current_mode == 'BLIND_AUDIT' else f'run_{current_mode.lower()}.csv'"
    if old_fname in src:
        src = src.replace(old_fname, new_fname)
        log(f"  OK  Cell {idx} (Export CSV): CSV file name adapted.")

    nb["cells"][idx]["source"] = [src]


def _patch_all_plt_show(nb):
    """
    Global pass: replaces all the remaining plt.show() calls with headless save+close.
    Smart handling: if the cell already explicitly saves its figures via .savefig() or is
    displaying an on-screen preview of a previously saved figure (plt.imshow), it replaces
    plt.show() with a clean plt.close() to prevent duplicate or duplicate-named images.
    Only cells that genuinely lack explicit figure saving will generate a fallback image
    inside NOTEBOOK_IMAGES_DIR.
    """
    counter = [0]
    n_saved_close = 0
    n_only_close = 0

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "plt.show()" not in src:
            continue

        cell_lines = src.splitlines()
        lines_out = []
        
        # Check if the entire cell has explicit savefig calls
        has_cell_savefig = any(".savefig(" in line for line in cell_lines)

        for l_idx, l in enumerate(src.splitlines(keepends=True)):
            if l.rstrip().endswith("plt.show()"):
                counter[0] += 1
                indent = len(l) - len(l.lstrip())
                pad = " " * indent
                ending = "\n" if l.endswith("\n") else ""
                
                # Check recent preceding non-comment lines
                preceding = [cell_lines[k].strip() for k in range(max(0, l_idx - 15), l_idx)
                             if cell_lines[k].strip() and not cell_lines[k].strip().startswith("#")]
                
                is_display_preview = any("plt.imshow" in line or "imshow(" in line for line in preceding)
                is_preceded_by_save = any(".savefig(" in line for line in preceding)
                
                if has_cell_savefig or is_display_preview or is_preceded_by_save:
                    # Cell already manages saving or is just showing a preview: only close to avoid duplicates
                    replacement = f"{pad}plt.close()"
                    n_only_close += 1
                else:
                    # Cell creates a figure but never saves it explicitly: save fallback to NOTEBOOK_IMAGES_DIR
                    replacement = _plt_show_replacement(indent, idx, counter[0])
                    n_saved_close += 1

                lines_out.append(replacement + ending)
            else:
                lines_out.append(l)

        nb["cells"][idx]["source"] = ["".join(lines_out)]

    log(f"  OK  Global plt.show() pass: {n_only_close} closed without duplicate, {n_saved_close} fallback saved.")


def build_patched_notebook(source_path, out_path):
    """Phase 1: patches the critical cells and saves the intermediate notebook."""
    global CELL_SECTION_NAMES
    global CELL_INDEX_SETUP, CELL_INDEX_MASTER_CFG, CELL_INDEX_MODEL_ARCH
    global CELL_INDEX_INFERENCE, CELL_INDEX_EXPORT_CSV

    log("")
    log("=" * 70)
    log(f"PHASE 1 - BUILD  |  {source_path}")
    log("=" * 70)

    with open(source_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    log(f"  Total cells: {len(nb['cells'])}")

    # --- AUTO-DETECT: section names map and critical cell indices ---
    CELL_SECTION_NAMES = _build_section_names_map(nb)
    log(f"  OK  Auto-detected {len(CELL_SECTION_NAMES)} sections for image naming.")

    detected = _autodetect_cell_indices(nb)
    CELL_INDEX_SETUP      = detected['setup']
    CELL_INDEX_MASTER_CFG = detected['master_cfg']
    CELL_INDEX_MODEL_ARCH = detected['model_arch']
    CELL_INDEX_INFERENCE  = detected['inference']
    CELL_INDEX_EXPORT_CSV = detected['export_csv']
    log(f"  OK  Auto-detected cell indices:")
    log(f"       SETUP={CELL_INDEX_SETUP}  MASTER_CFG={CELL_INDEX_MASTER_CFG}  "
        f"MODEL_ARCH={CELL_INDEX_MODEL_ARCH}  INFERENCE={CELL_INDEX_INFERENCE}  "
        f"EXPORT_CSV={CELL_INDEX_EXPORT_CSV}")
    # ---------------------------------------------------------------

    _patch_cell_setup(nb, CELL_INDEX_SETUP)
    _patch_cell_master_config(nb, CELL_INDEX_MASTER_CFG)
    _patch_cell_model_arch(nb, CELL_INDEX_MODEL_ARCH)
    _patch_cell_inference(nb, CELL_INDEX_INFERENCE)
    _patch_cell_export_csv(nb, CELL_INDEX_EXPORT_CSV)
    _patch_all_plt_show(nb)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    log(f"\n  >> Patched notebook saved: {out_path}")
    return nb


# ==============================================================================
# PHASE 2 - CONVERT: notebook -> .py script
# ==============================================================================

def convert_notebook_to_script(nb_path, py_path, source_nb_name):
    """Phase 2: converts the patched notebook into a standalone Python script."""
    log("")
    log("=" * 70)
    log(f"PHASE 2 - CONVERT  |  {nb_path} -> {py_path}")
    log("=" * 70)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "#!/usr/bin/env python",
        "# -*- coding: utf-8 -*-",
        '"""',
        "Unified Benchmark & XAI Forensic Diagnostics Framework",
        "Standalone Version Runnable from the Terminal (CLI & Headless-Ready)",
        f"Derived from: {source_nb_name}",
        f"Automatically generated by sync_framework.py on {now}",
        '"""',
        "",
        "# Headless matplotlib - MUST precede any other import",
        "import os",
        "import sys",
        "import logging",
        "import matplotlib",
        "try:",
        '    matplotlib.use("Agg")',
        "except Exception:",
        "    pass",
        "",
    ]

    cells_converted = 0
    for idx, cell in enumerate(nb["cells"]):
        ctype = cell["cell_type"]
        source = cell["source"]
        text = "".join(source) if isinstance(source, list) else str(source)

        if ctype == "markdown":
            lines.append("")
            lines.append("# " + "=" * 78)
            lines.append(f"# [CELL {idx:02d} - MARKDOWN]")
            for l in text.splitlines():
                lines.append(f"# {l}")
            lines.append("# " + "=" * 78)

        elif ctype == "code":
            lines.append("")
            lines.append("# " + "=" * 78)
            lines.append(f"# [CELL {idx:02d} - CODE]")
            lines.append("# " + "=" * 78)

            for l in text.splitlines():
                s = l.strip()
                if s.startswith("!") or s.startswith("%"):
                    lines.append(f"# [IPYTHON MAGIC] {l}")
                elif "from IPython.display import" in s:
                    lines.append(f"# [IPYTHON IMPORT SKIPPED] {l}")
                elif s.startswith("display(") and not s.startswith("display =") and not s.startswith("display_"):
                    lines.append(l.replace("display(", "print(", 1))
                else:
                    lines.append(l)

            lines.append("")
            cells_converted += 1

    with open(py_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log(f"\n  >> .py script saved: {py_path}")
    log(f"     Code cells transcribed: {cells_converted} / {len(nb['cells'])}")

    import py_compile
    try:
        py_compile.compile(py_path, doraise=True)
        log("  OK  Python syntax check: no errors.")
    except py_compile.PyCompileError as e:
        log(f"  SYNTAX ERROR: {e}")
        log("  Check the generated file before running it.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    log("")
    log("=" * 72)
    log("  sync_framework.py  --  Notebook -> Standalone CLI Script")
    log("=" * 72)
    log(f"  Source        : {SOURCE_NOTEBOOK}")
    log(f"  Output dir    : {OUTPUT_DIR_FRAMEWORK}/")
    log(f"  Notebook patch: {OUTPUT_DIR_FRAMEWORK}/{OUTPUT_BASENAME}.ipynb")
    log(f"  Script Python : {OUTPUT_DIR_FRAMEWORK}/{OUTPUT_BASENAME}.py")
    log("")

    if not os.path.exists(SOURCE_NOTEBOOK):
        log(f"ERROR: source notebook not found: '{SOURCE_NOTEBOOK}'")
        log("Check the value of SOURCE_NOTEBOOK in sync_framework.py.")
        sys.exit(1)

    patched_nb = os.path.join(OUTPUT_DIR_FRAMEWORK, OUTPUT_BASENAME + ".ipynb")
    output_py  = os.path.join(OUTPUT_DIR_FRAMEWORK, OUTPUT_BASENAME + ".py")

    t0 = datetime.datetime.now()

    build_patched_notebook(SOURCE_NOTEBOOK, patched_nb)
    convert_notebook_to_script(patched_nb, output_py, SOURCE_NOTEBOOK)

    elapsed = (datetime.datetime.now() - t0).total_seconds()

    log("")
    log("=" * 72)
    log(f"  COMPLETED in {elapsed:.1f}s")
    log("=" * 72)
    log(f"  Patched notebook  : {patched_nb}")
    log(f"  Standalone script : {output_py}")
    log("")
    log("  Run with:")
    log(f"    python {output_py} \\")
    log("      --dataset_path <dataset_folder> \\")
    log("      --model_path <checkpoint.pth> \\")
    log("      --mode blind_audit")
    log("")


if __name__ == "__main__":
    main()
