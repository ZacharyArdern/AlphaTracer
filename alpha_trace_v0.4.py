import polars as pl
import parasail
from polars_strsim import levenshtein
import re
import plotext as plt
import os
import time
import argparse


# --- Setup ---

pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_rows(10)
pl.Config.set_fmt_str_lengths(25)
pl.Config.set_tbl_hide_dataframe_shape(True)
plt.theme('elegant')

ascii_art1 = r"""
     _    _     ____  _   _    _  _____ ____      _    ____ _____ ____  
    / \  | |   |  _ \| | | |  / \|_   _|  _ \    / \  / ___| ____|  _ \ 
   / _ \ | |   | |_) | |_| | / _ \ | | | |_) |  / _ \| |   |  _| | |_) |
  / ___ \| |___|  __/|  _  |/ ___ \| | |  _ <  / ___ \ |___| |___|  _ < 
 /_/   \_\_____|_|   |_| |_/_/   \_\_| |_| \_\/_/   \_\____|_____|_| \_\
"""
ascii_art2 = r"""
        _       _                 _                                     
   __ _| |_ __ | |__   __ _      | |_ _ __ __ _  ___ ___                
  / _` | | '_ \| '_ \ / _` |_____| __| '__/ _` |/ __/ _ \               
 | (_| | | |_) | | | | (_| |_____| |_| | | (_| | (_|  __/               
  \__,_|_| .__/|_| |_|\__,_|      \__|_|  \__,_|\___\___|               
         |_|                                                            
"""
# ASCII art from: https://patorjk.com/software/taag/#p=display&f=Ivrit&t=ALPHATRACER%0Aalpha-trace


parser = argparse.ArgumentParser(description="Divide hits into similarity classes")

parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input fasta file of query sequences')

parser.add_argument('--doctest', action='store_true',
                        help='Run doctests implemented in this script')

args = parser.parse_args()

input_basename = os.path.basename(args.input).split('.')[0]
processing_dir = f"AT_processing_{input_basename}"


###################################################################

##################  --- Helper Functions ---  #####################

###################################################################


def align_row_nw(row):
    """Align two seqs using parasail needleman-wunsch (nw).
    Usage example:
    >>> align_row_nw(("QID", "SID", "MLK", "LLKE"))
    ('QID', 'SID', 'MLK-', 'LLKE')
    """
    try:
        qseqid, sseqid, seq1, seq2 = row  # unpack the tuple
        ali = parasail.nw_trace_striped_16(seq1, seq2, 10, 1, parasail.blosum45)
        # print('ALI TRACEBACK QUERY', '\n', ali.traceback.query, sep='')
        # print('ALI TRACEBACK REF', '\n', ali.traceback.ref, sep='')
        # print('ALI TRACEBACK COMP', '\n', ali.traceback.comp, sep='')
        comp_str = ali.traceback.comp.replace(" ", "-")
        return qseqid, sseqid, ali.traceback.query, ali.traceback.ref, comp_str
    except Exception:
        return []

def fix_hanging_group_letters(seq: str) -> str:
    """If one, two, or three ntds are at the start or end of an alignment followed/preceded by gaps, then shift the gaps to start/end.
    Usage examples:
    >>> test_cases = [ ("A----TGC", "----ATGC"), ("A--TGCCG", "A--TGCCG"), ("ATGGTT---C", "ATGGTTC---") ]
    >>> for inp, expected in test_cases:
    ...     assert fix_hanging_group_letters(inp) == expected
    """
    # Left side: 1–3 letters followed by 3+ dashes then rest of seq
    # Left side: 1–3 non-dash characters followed by 3+ dashes
    left_match = re.match(r"^([^-]{1,3})(-+)", seq)
    if left_match and len(left_match.group(2)) >= 3:
        letters, dashes = left_match.groups()
        seq = ("-" * len(dashes)) + letters + seq[len(letters) + len(dashes):]

    # Right side: 3+ dashes followed by 1–3 non-dash characters at end
    right_match = re.search(r"(-+)([^-]{1,3})$", seq)
    if right_match and len(right_match.group(1)) >= 3:
        dashes, letters = right_match.groups()
        seq = seq[:-(len(dashes) + len(letters))] + letters + ("-" * len(dashes))

    return seq


def chunk_match_proportions(s: str, size: int = 20) -> dict[tuple[int, int], float]:
    if not isinstance(s, str): return {}
    result, start, count, buf = {}, 0, 0, []
    for i, c in enumerate(s):
        buf.append((i, c))
        count += c != '-'
        if count == size:
            indices, chars = zip(*buf)
            non_dashes = [ch for ch in chars if ch != '-']
            matches = sum(1 for ch in non_dashes if ch in '|:')
            result[(indices[0], indices[-1] + 1)] = matches / size
            buf, count = [], 0
    if count:
        indices, chars = zip(*buf)
        non_dashes = [ch for ch in chars if ch != '-']
        matches = sum(1 for ch in non_dashes if ch in '|:')
        result[(indices[0], indices[-1] + 1)] = matches / size
    return result


plt.colorize(f"{ascii_art1}\n© ZACHARY ARDERN 2025\ncomments to: z.ardern@gmail.com or https://github.com/ZacharyArdern/AlphaTracer/", "blue", "bold", "-", True)

print( 
    "\n\n",
    "RUNNING MODULE 2; DIVIDES SEQUENCES INTO CLASSES BY HOMOLOGOUS SEQUENCE TYPE",
    f"{ascii_art2}" + "\n" * 2,
    sep="",
)


###################################################################

#########  DEFINE CLASSIFYING FUNCTIONS FOR EACH SECTION  #########

###################################################################

### PRE-PROCESSING

def preprocess_hits(hits_file):
    tophits_df = pl.read_parquet(hits_file).drop('evalue')
    preprocessed_hits_df = tophits_df.filter(~pl.col("full_qseq").str.contains("X"))
    seqs_filtered_by_X = len(tophits_df) - len(preprocessed_hits_df)
    print("*****", "\n", "NOTE: ", seqs_filtered_by_X, " sequences excluded due to containing 'X' ", '\n', "*****", "\n\n", sep='')
    num_seqs_total = len(preprocessed_hits_df)
    print("INPUT PARQUET FILE FROM ALPHA_SEARCH", "\n", 'seqs: ', len(preprocessed_hits_df), "\n", preprocessed_hits_df, "\n\n", sep='')
    return preprocessed_hits_df 


### CLASS A - near-identical, no gaps - take full co-ordinates
# FIND CLOSE HITS OF CORRECT LENGTH NOT REQUIRING ALIGNMENT - COORDINATES FROM CORRESPONDING SSEQS CAN BE USED DIRECTLY

# (to run with preprocessed_hits_df)
def get_classA(input_df):
    classA_df = input_df.filter(
        (pl.col("approx_pident") >= 80) & (pl.col("slen") == pl.col("qlen"))
    ).drop(['qlen', 'slen'])
    hits_notclassA_df  = input_df.join(classA_df, on="qseqid", how="anti").drop(['qlen', 'slen'])

    print("CLASS A: very close hits", "\n", 'seqs: ', len(classA_df), '\n', classA_df, "\n\n", sep='')
    classA_df.write_parquet(os.path.join(processing_dir, 'classA.pq'))
    return classA_df, hits_notclassA_df 

#####################

### CLASS B - very similar, no gaps but can be embedded (substring) - take subset of co-ordinates

# (run with hits_notclassA_df )
def get_classB(input_df):
    # ALIGN SEQUENCES REQUIRING ALIGNING  # Apply alignment function using map_rows to get new_cols and add new_cols to the dataframe

    alignedseqs_subdf = (
        hits_notclassA_df.select(["qseqid", "sseqid", "full_qseq", "full_sseq"])
        .map_rows(align_row_nw, return_dtype=pl.List(pl.String))
    ).rename( {'column_0':'qseqid', 'column_1':'sseqid', 'column_2':'qseq_alg', 'column_3':'sseq_alg', 'column_4':'alg_comp'}).with_columns(
        ( (pl.col("alg_comp").str.count_matches(r"[|:]")) / 
         (pl.col("alg_comp").str.count_matches(r"[^-]")) * 100).alias("%pos"))

    # note that the %pos excludes any gaps including gaps in sseq relative to qseq

    # print('alignedseqs_subdf', '\n', alignedseqs_subdf, sep='')

    hits_notclassA_alg_df = hits_notclassA_df.join(
    alignedseqs_subdf,
    on=["qseqid", "sseqid"],
    how="inner" )

    hits_notclassA_alg_df = hits_notclassA_alg_df.with_columns(
        [pl.col("qseq_alg").map_elements(fix_hanging_group_letters, return_dtype=pl.String)]
    ).with_columns(
        [pl.col("sseq_alg").map_elements(fix_hanging_group_letters, return_dtype=pl.String)]
    ).with_columns(
        [pl.col("alg_comp").map_elements(fix_hanging_group_letters, return_dtype=pl.String)]
    )


    # # FIND QSEQS FULLY 'EMBEDDED' WITHIN SSEQS - CO-ORDINATES FROM CORRESPONDING SSEQS CAN ALSO BE USED DIRECTLY
    # # for q seq: only start/end dashes & no dashes in the middle; for s seq no dashes are permitted
    classB_df = hits_notclassA_alg_df.filter(
        (pl.col("qseq_alg").str.contains(r"^-*[^-]+-*$"))
        & (~pl.col("qseq_alg").str.contains(r"[^-]-+[^-]"))
        & (~pl.col("sseq_alg").str.contains("-"))
    ).filter((pl.col("%pos")>=70))   

    hits_notclassAB_df = hits_notclassA_alg_df.join(
        classB_df, on="qseqid", how="anti").drop(['full_qseq', 'full_sseq'])

    print("CLASS B: embedded hits", "\n", 'seqs: ', len(classB_df), '\n', classB_df, "\n\n", sep='')
    classB_df.write_parquet(os.path.join(processing_dir, 'classB.pq'))

    # print('hits_notclassAB_df', '\n', hits_notclassAB_df, sep='')

    return classB_df, hits_notclassAB_df 


#####################

### CLASS C - some minor gaps present in sseq_alg relative to qseq_alg, structure inference not required - can take angles and infer with int2cart
# note, previous versions of this script (<=0.3) included using multiple homologs, but this adds significant complexity - may be re-added later

def get_classC(input_df):
    
    # modify alignment trace to indicate which aligned seq has the gap 
    # (- if query [missing in query], ~ if subject [missing in AF subject, needs to be inferred]): 
    input_df2 = input_df.with_columns(
    pl.struct(['sseq_alg', 'alg_comp']).map_elements(
        lambda row: ''.join('~' if s == '-' else a for s, a in zip(row['sseq_alg'], row['alg_comp'])), return_dtype=pl.String
    ).alias('alg_comp') )

    classC_df = input_df2.with_columns(
    pl.col("sseq_alg").str.count_matches("-").alias("sseq_dash_count")
    ).filter(( pl.col("sseq_dash_count") <=2) & (pl.col("%pos") >=70 ))

    print("CLASS C: minor gaps in subject sequence relative to query: ", "\n", 'seqs: ', len(classC_df), '\n', classC_df, "\n", sep='')

    classC_df.write_parquet(os.path.join(processing_dir, 'classC.pq'))
    hits_notclassABC_df = hits_notclassAB_df.join(classC_df, on='qseqid', how='anti')

    return classC_df, hits_notclassABC_df 


#####################

##### CLASS D seqs to infer structures for
# split into PAE-based domains
# find regions of the query sequences which are not adequately similar to the homolog 
# i.e. windows around gaps in sseq or low match windows

def get_classD_seqs(input_df):

    # modify alignment trace to indicate which seq has the gap (- if query, ~ if subject): 
    hits_D_df = input_df.with_columns(
    pl.struct(['sseq_alg', 'alg_comp']).map_elements(
        lambda row: ''.join('~' if s == '-' else a for s, a in zip(row['sseq_alg'], row['alg_comp'])), return_dtype=pl.String
    ).alias('alg_comp') )

    # # iterate over sequence windows (length 20aa's), across modified comp_alg - skip gaps in qseq ("-")
    hits_D_df = hits_D_df.with_columns(
    pl.col('alg_comp').map_elements(
        lambda s: chunk_match_proportions(s), return_dtype=pl.Object
    ).alias('match_proportions'))

    # classify *alg indices* into close_homolog and not_close_homolog (these can later? be extracted from qseq or torsion angles)

    # Convert dict to list of structs, explode, and unpack keys and values
    hits_D_df_windows = hits_D_df.with_columns(
        pl.col("match_proportions").map_elements(
            lambda d: [{"key": k, "value": v} for k, v in d.items()] if d else [] 
        ).alias("kv_list") )
    
    print('test', '\n', hits_D_df_windows, sep='')

    hits_D_df_windows = hits_D_df_windows.explode("kv_list").with_columns([
        pl.col("kv_list").struct.field("key").alias("alg_chunk_idx"),
        pl.col("kv_list").struct.field("value").alias("proportion_pos")
    ]).drop("kv_list")

    hits_D_df_windows = hits_D_df_windows.with_columns(
    pl.struct(['qseq_alg', 'alg_chunk_idx']).map_elements(
        lambda x: x['qseq_alg'][x['alg_chunk_idx'][0]:x['alg_chunk_idx'][1]],
        return_dtype=pl.String
    ).alias('qseq_alg_chunk')
    ).drop(['qseq_alg', 'sseq_alg', 'alg_comp', 'approx_pident', '%pos', 'match_proportions']
    ).with_columns(
    pl.col('qseq_alg_chunk').str.replace_all('-', '').alias('qseq_alg_chunk_clean')
    )

    print('hits_D_df_windows', '\n', hits_D_df_windows, sep='')

    hits_D_df_windows.write_parquet(os.path.join(processing_dir, 'ClassD_windows.pq'))


    merged_dfs_list = []

    for name, data in hits_D_df_windows.group_by(['qseqid', 'sseqid']):
        test_df = data.with_columns([
            (pl.col('proportion_pos') >= 0.8).alias('is_good'),
            (pl.col('proportion_pos').shift(1) >= 0.8).fill_null(True).alias('prev_good'),
            (pl.col('proportion_pos').shift(-1) >= 0.8).fill_null(True).alias('next_good'),
            pl.when(pl.col('proportion_pos') >= 0.8)
            .then(
                pl.when((pl.col('proportion_pos').shift(1) >= 0.8).fill_null(True) & 
                        (pl.col('proportion_pos').shift(-1) >= 0.8).fill_null(True))
                .then(pl.lit('close_homolog'))
                .otherwise(pl.lit('buffer'))
            )
            .otherwise(pl.lit('not_close'))
            .alias('label')
        ])

        test_df = test_df.with_columns(
            ((pl.col('label') == 'buffer') | (pl.col('label') == 'not_close')).alias('to_infer')
        )

        test_df = test_df.with_columns(
            (pl.col('to_infer') != pl.col('to_infer').shift(1)).fill_null(True).alias('boundary')
        )
        test_df = test_df.with_columns(
            pl.col('boundary').cum_sum().alias('group')
        )

        # Step 4: Filter groups where keep=True (buffer or to_infer)
        valid_groups = test_df.filter(pl.col('to_infer')).select('group').unique()
        df_valid = test_df.join(valid_groups, on='group')

        # print('df_valid', '\n' , df_valid, sep='')

        # note - need to keep "group"s separate below as each 'group' is a consecutive block of windows which should be merged

        merged_df_x = (
            df_valid
            .with_columns([
                pl.col('alg_chunk_idx').list.get(0).alias('start_idx'),  # extract start
                pl.col('alg_chunk_idx').list.get(1).alias('end_idx'),    # extract end
            ])
            .group_by(['qseqid', 'sseqid', 'group'])
            .agg([
                pl.col('start_idx').min().alias('start_idx'),
                pl.col('end_idx').max().alias('end_idx'),
                pl.col('qseq_alg_chunk_clean').implode()
                .map_elements(lambda x: ''.join(x), return_dtype=pl.Utf8)
                .alias('merged_sequence_clean'),
                pl.len().alias('num_windows')
            ])
            .with_columns([
                pl.concat_list(['start_idx', 'end_idx']).alias('merged_range_alg')  # create list column correctly
            ])
            .select(['qseqid', 'sseqid', 'merged_range_alg', 'merged_sequence_clean'])
        )

        merged_dfs_list.append(merged_df_x)

        # print('merged_df_x', ' ', name,  '\n', merged_df_x, sep='')

    merged_df = pl.concat(merged_dfs_list)
    print('merged_df',  '\n', len(merged_df), '\n', merged_df, sep='')

    df = hits_D_df_windows

    df = df.with_columns(
        (pl.col('proportion_pos') < 0.8).alias('keep')
    ).with_columns(
        (pl.col('keep') != pl.col('keep').shift(1)).fill_null(True).alias('boundary')
    ).with_columns(
        pl.col('boundary').cum_sum().alias('group')
    )

    valid_groups = df.filter(pl.col('keep')).select('group').unique()

    df_valid = df.join(valid_groups, on='group')

    merged_to_infer = (
        df_valid
        .group_by(['qseqid', 'sseqid', 'group'])
        .agg([
            pl.col('alg_chunk_idx').list.get(0).first().alias('start_idx'),
            pl.col('alg_chunk_idx').list.get(1).last().alias('end_idx'),
            pl.col('qseq_alg_chunk_clean').implode().map_elements(''.join, return_dtype=pl.String).alias('merged_sequence_clean')
        ])
        .with_columns([
            pl.struct(['start_idx', 'end_idx']).map_elements(
                lambda s: [s['start_idx'], s['end_idx']],
                return_dtype=pl.List(pl.Int64)
            ).alias('merged_range_alg')
        ])
        .select(['qseqid', 'sseqid', 'merged_range_alg', 'merged_sequence_clean'])
    )


    merged_to_infer = merged_to_infer.select(['qseqid', 'merged_range_alg', 'merged_sequence_clean']).with_columns(
            pl.col('merged_range_alg')
            .map_elements(lambda x: f'{x[0]}:{x[1]}', return_dtype=pl.String)
            .alias('merged_range')
        ).drop(['merged_range_alg'])
    
    print('merged_to_infer', '\n', merged_to_infer, sep='')

    merged_to_infer.write_csv(os.path.join(processing_dir, 'classD_seqstoinfer.csv'), include_header=False, separator='\t')

    return hits_D_df_windows


# #####################

# #### CLASS E - no matching regions at all, need to infer full structure

def get_classE_seqs():
    headers = []
    sequences = []

    with open(os.path.join(processing_dir, 'input_seqs_filtered.fa')) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            header = lines[i].strip()
            sequence = lines[i + 1].strip()
            if header.startswith(">"):
                headers.append(header[1:])  # remove the '>'
                sequences.append(sequence)
            else:
                raise ValueError(f"Unexpected line format at line {i}: {header}")

    # Create Polars DataFrame
    inputseqs_df = pl.DataFrame({
        "qseqid": headers,
        "sequence": sequences
    })

    classE_seqs_df = inputseqs_df.join(AtoD_seqs_df, on='qseqid', how='anti')

    num_E_seqs = len(classE_seqs_df)

    print(len(classE_seqs_df))
    print(classE_seqs_df, '\n\n')


    classE_seqs_df.write_csv(os.path.join(processing_dir, 'classE_seqs.txt'), include_header=False, separator='\t')

    return classE_seqs_df  


########################

# RUN CLASSIFYING FUNCTIONS:
# if a "remaining" df is empty, skip remaining classifying functions until class E

empty_df = pl.DataFrame(schema=[('qseqid', pl.String)])

# PRE-PROCESSING

preprocessed_hits_df = preprocess_hits(os.path.join(processing_dir, 'afdb_tophits.pq')) 

# CLASS A
classA_df, hits_notclassA_df = get_classA(preprocessed_hits_df)

# CLASS B
if len(hits_notclassA_df) >= 0:
    classB_df, hits_notclassAB_df = get_classB(hits_notclassA_df)
else:
   classB_df, hits_notclassAB_df = empty_df, empty_df

# CLASS C
if len(hits_notclassAB_df) >= 0:
    classC_df, hits_notclassABC_df = get_classC(hits_notclassAB_df)
else:
    classC_df, hits_notclassABC_df = empty_df, empty_df

# print('hits_notclassABC_df', '\n', hits_notclassABC_df, sep='')

# CLASS D

if len(hits_notclassABC_df) >= 0:
    classD_df = hits_notclassABC_df 
    classD_df.write_parquet(os.path.join(processing_dir,'classD.pq'))
    hits_D_df_windows = get_classD_seqs(hits_notclassABC_df)


AtoD_seqs_df = pl.concat([classA_df.select('qseqid').unique(), 
                        classB_df.select('qseqid').unique(), 
                        classC_df.select('qseqid').unique(), 
                        classD_df.select('qseqid').unique() ])
print('AtoD_seqs_df', '\n', len(AtoD_seqs_df), '\n', AtoD_seqs_df, sep='')


# CLASS E

classE_seqs_df = get_classE_seqs()


# ######################
# # WRITE SEQUENCES THAT NEED TO HAVE STRUCTURES INFERRED TO A NEW FASTA FILE

# classE_seqstoinfer_df = classE_noadequatehomologs_df.select(['qseqid', 'window_number', 'qseq_window'])

# with open(os.path.join(processing_dir, 'seqs_to_infer_E.fasta'), "w") as f:
#     for row in classE_seqstoinfer_df.iter_rows():
#         f.write(f">{row[0]}:{row[1]}\n{row[2]}\n")

# with open(os.path.join(processing_dir, 'seqs_to_infer_F.fasta'), "w") as f:
#     for row in classF_seqs_df.iter_rows():
#         f.write(f">{row[0]}:FULL\n{row[1]}\n")



# # ######################
# # # FINAL PLOT TO SHOW DISTRIBUTION ACROSS CLASSES

num_A_seqs = len(classA_df)
num_B_seqs = len(classB_df)
num_C_seqs = len(classC_df)
num_D_seqs = len(hits_D_df_windows.select('qseqid').unique())
num_E_seqs = len(classE_seqs_df.select('qseqid').unique())



W,H = plt.terminal_size()

plt.plot_size(W/2, H/2)
classes = ["A", "B", "C", "D", "E", "F"]
percentages = [ num_A_seqs, 
               num_B_seqs, 
               num_C_seqs, 
               num_D_seqs,
               num_E_seqs
                      ]

plt.bar(classes, percentages)
plt.axes_color('indigo')
plt.title("Number of Sequences by Homology Class")
plt.show()


print(
    "\n",
    "COMPLETED MODULE 2; DIVIDED SEQUENCES INTO CLASSES BY HOMOLOGOUS SEQUENCE TYPE.",
    "\n",
    sep="\n",
)

# exit("Stopped for debugging")

if args.doctest:

    import doctest

    print(
        "\n",
        "######",
        "\n",
        "#######",
        "\n",
        "REPORTING DOCTESTS:",
        "\n",
        "#######",
        "\n",
        "######",
        "\n",
        sep="",
    )

    doctest.testmod(verbose=True)

exit("Program concluded")





