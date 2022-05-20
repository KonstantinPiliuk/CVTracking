from Detector import Detector
from Tracker import Tracker
import argparse

def init_args():
    parser = argparse.ArgumentParser(description='Arguments for tracking process')
    parser.add_argument("--vid", required=True, type=str, help="Video file for computation")
    parser.add_argument("--w", required=True, default='inputs/coords_model.pt', type=str, help="Model weights file for keypoints detection")
    parser.add_argument("--log", required=True, default='output.csv', type=str, help="Log to SQL Database or csv file")
    parser.add_argument("--d", default=None, type=str, help="SQL dialect")
    parser.add_argument("--h", default=None, type=str, help="Host adress for SQL")
    parser.add_argument("--db", default=None, type=str, help="Database name for SQL")
    parser.add_argument("--u", default=None, type=str, help="Username for SQL")
    parser.add_argument("--p", default=None, type=str, help="Password for SQL")
    return parser

def main():
    parser = init_args()
    args = parser.parse_args()
    a = Detector(vid=args.vid, weights=args.w, log=args.log, dialect=args.d, host=args.h, user=args.u, pwd=args.p, database=args.db)
    a.funnel()
    b = Tracker(log=args.log, dialect=args.d, host=args.h, user=args.u, pwd=args.p, database=args.db)
    b.fit(a.output)

if __name__== "__main__":
    main()