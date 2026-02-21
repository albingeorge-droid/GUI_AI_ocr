import argparse
import subprocess
import sys
from typing import List


BUCKET_NAME = "guai"
REGION = "us-east-2"
PROFILE = "default"

# --- Hardcoded lists ---
ARTIFACTS_0_100 = [
    # "3202_baljeet_singjh_matharu_8-60wpb/",
    # "3203_girish_kumar_26epb/",
    # "3204_sapna_jain_12-32epb/",
    # "3205_arvinder_kaur_10-47wpb/",
    # "3206_harish_chander_chhabra_25-42wpb/",
    # "3207_manju_kohli_13-33epb/",
    # "3208_pushpa_oswal_8-33epb/",
    # "3212_urmila_chouwdhary_25-17epb/",
    # "3213_sjurinder_mohan_bhatia_17epb/",
    # "3213_suraj_parkash_17epb/",
    # "3214_manmohan_bhasin_27-44wpb/",
    # "3218_meeta_rani_bhalla_86epb/",
    # "3220_gurpreet_singh_7-65wpb/",
    # "3225_sushil_sabharwal_26-23epb/",
    # "3231_krishan_gopal_kalra_5-63wpb/",

    # "3162_deepak_mehta_6-62wpb/",
    # "3167_vimla_goyal_7-17epb/",
    # "3169_komal_gulati_21-62wpb/",
    # "3170_rajeshwari_bhandari_13-72wpb/",
    # "3171_surender_kumar_12-46wpb/",
    # "3173_santosh_batra_32-66wpb/",
    # "3177_harinder_kohli_5-33-epb/",
    # "3190_mukesh_marwaha_14-32epb/",
    # "3192_amandeep_singh_sehgal_10-46wpb/",
    # "3195_vijay_prabhakar_a29-42wpb/",
    # "3197_prehlad_singh_sethi_22-66wpb/",
    # "3198_virender_kumar_sehgal_19epb/",
    # "3199_kewal_krishan_sood_12-4epb/",
    "4658_CLU_RK-994/",
    "2351_CLU_GN-1778/",
    "2223_CLU_GN-2511/",
    "7432_CLU_PT-1719A/",


]

ARTIFACTS_100_200 = [
    # "3201_mahavir_goel_8-22epb/",
    # "3211_kunal_khanna_18-63wpb/",
    # "3219_vinay_kumar_sehgal_23-27epb/",
    # "3222_sachin_agarwal_22-60wpb/",
    # "3232_satish_gupta_26-44wpb/",

    # "3164_parveen_bala_3-40wpb/",
    # "3166_illa_mehra_e44wpb/",
    # "3172_yashpal_gupta_29-5epb/",
    # "3174_santosh_kumar_chhabra_35wpb/",
    # "3191_yespal_mendiratta_10-42wpb/",
    # "3370_ajay_makkar_1-26epb/",

    # "3237_anil_nandra_17-63wpb/",
    # "3338_vibhu_aggarwal_20-26epb/",
    # "3340_h_r_polycoats_pvt_ltd_10-60wpb/",
    # "3341_rakesh_goel_1-8epb/",
    # "3344_kusam_aggarwal_16b-12epb/",
    # "3345_devender_k_narang_3-63wpb/",
    # "3357_deepak_kumar_24-64wpb/",
    # "3358_nirmal_gupta_21-32epb/",
    # "3366_abhay_gahlot_41-35wpb/",
    # "3368_meenu_goel_38-7epb/",
]

ARTIFACTS_200_PLUS = [
    # "3163_aman_gupta_4-23-epb/",
    # "3165_varun_khurana_51-42-wpb/",
    # "3175_anita_sachdeva_15-47wpb/",
    # "3176_daljit_singh_bindra_35-64wpb/",
    # "3178_vivek_jain_4-22epb/",
    # "3194_ravi_kochhar_19-16epb/",

    # "3339_surender_kumar_bansal_17-77wpb/",
    # "3346_sunil_singhal_47-78wpb/",
    # "3351_asha_mundra_9-84wpb/",
    # "3352_meenakshi_devi_1epb/",
    # "3356_rekha_sharma_77-7epb/",
    # "3360_mahender_p_malhotra_4-65wpb/",
    # "3363_arun_jindal_12-53bwpb/",
]

# --- Roots ---
ROOTS = [
    # "artifacts_29_07_2025",
    # "artifacts_26_07_2025",
    "Haryana1",  # keep empty root as in your PS script
]


def run_aws(args: List[str], dry_run: bool) -> int:
    """
    Runs an AWS CLI command and streams output live.
    Returns the process exit code.
    """
    full_cmd = ["aws"] + args + ["--region", REGION, "--profile", PROFILE]
    print("Running:", " ".join(full_cmd))

    if dry_run:
        return 0

    # Stream output live (stdout + stderr)
    proc = subprocess.Popen(
        full_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        shell=False,
    )
    return proc.wait()


def s3_prefix_exists(s3_uri: str, dry_run: bool) -> bool:
    """
    Checks whether an S3 prefix exists by running: aws s3 ls <prefix>
    In dry-run mode, we assume it exists (to avoid blocking simulation).
    """
    if dry_run:
        print(f"(dryrun) would check: aws s3 ls {s3_uri} --region {REGION} --profile {PROFILE}")
        return True

    cmd = ["aws", "s3", "ls", s3_uri, "--region", REGION, "--profile", PROFILE]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # If exit code is non-zero or output empty, treat as not found
    return (result.returncode == 0) and (result.stdout.strip() != "")


def move_folders(dest: str, folders: List[str], move: bool, dry_run: bool) -> None:
    for r in ROOTS:
        for f in folders:
            # Build source and destination exactly like your PS logic
            if r:
                src = f"s3://{BUCKET_NAME}/{r}/{f}"
            else:
                src = f"s3://{BUCKET_NAME}/{f}"

            if dest:
                dst = f"s3://{BUCKET_NAME}/{dest}/{f}"
            else:
                dst = f"s3://{BUCKET_NAME}/{f}"

            print(f"\n→ {src} → {dst}")

            if not s3_prefix_exists(src, dry_run=dry_run):
                print(f"[WARNING] Source not found: {src}")
                continue

            code = run_aws(["s3", "sync", src, dst, "--exact-timestamps"], dry_run=dry_run)
            if code != 0:
                print("[AWS ERROR] sync failed, skipping delete.")
                continue

            if move and not dry_run:
                run_aws(["s3", "rm", src, "--recursive"], dry_run=dry_run)
            elif move and dry_run:
                print(f"(dryrun) would delete: aws s3 rm {src} --recursive --region {REGION} --profile {PROFILE}")


def main():
    parser = argparse.ArgumentParser(description="Move/Copy selected S3 folder prefixes using aws s3 sync.")
    parser.add_argument("--move", action="store_true", help="Delete source after successful copy (sync).")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions only (no aws commands executed).")
    args = parser.parse_args()

    # --- Execution (matches your PS script default) ---
    move_folders(dest="Haryana4", folders=ARTIFACTS_0_100, move=args.move, dry_run=args.dry_run)
    # move_folders(dest="artifacts_5-11_100-200", folders=ARTIFACTS_100_200, move=args.move, dry_run=args.dry_run)
    # move_folders(dest="artifacts_5-11_200-and", folders=ARTIFACTS_200_PLUS, move=args.move, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()


# Run from CMD
# python move_selected_files_folders.py


# Dry run:

# python move_selected_files_folders.py --dry-run


# Move (copy then delete source):

# python move_selected_files_folders.py --move


# Move + Dry run:

# python move_selected_files_folders.py --move --dry-run