import csv, os

INPUT_CSV = r"D:\hackamined\Copy of ICR2-LT1-Celestical-10000.73.raws.csv"
OUTPUT_DIR = r"D:\hackamined\parts"
CHUNK_SIZE = 20000

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_CSV, "r", newline="", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    header = next(reader)

    part = 1
    written = 0

    def open_part(p):
        path = os.path.join(OUTPUT_DIR, f"part_{p:03d}.csv")
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(header)
        return f, w

    out_f, w = open_part(part)

    for row in reader:
        if written >= CHUNK_SIZE:
            out_f.close()
            part += 1
            written = 0
            out_f, w = open_part(part)

        w.writerow(row)
        written += 1

    out_f.close()

print("Done.")