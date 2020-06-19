# hybridization-number-sat
Code for generating and testing the SAT formulation of the hybridization number

# Usage
`python3 pipeline.py <input file> [options]`

## Options

-h/--help   Display documentation.
-d  Print debug messages for where each variable starts.
-t  Time limit in seconds for the SAT solver. Default is 3600.
-o  Output directory for the results. If unspecified, results are only printed to stdout.
-n  Number of internal nodes to build the initial graph with. Default is n + m where n = number of rows, m = number of cols in input.
-b  Initial upper bound. Default is also n + m.
