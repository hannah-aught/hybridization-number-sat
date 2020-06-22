# hybridization-number-sat
Code for generating and testing the SAT formulation of the hybridization number

# Usage
`python3 pipeline.py <input file> [options]`

## Options

-h/--help   Display documentation. <br>
-d  Print debug messages for where each variable starts. <br>
-t  Time limit in seconds for the SAT solver. Default is 3600. <br>
-o  Output directory for the results. If unspecified, results are only printed to stdout. <br>
-n  Number of internal nodes to build the initial graph with. Default is n + m where n = number of rows, m = number of cols in input. <br>
-b  Initial upper bound. Default is also n + m. <br>
-s  Save the adjacency matrix for the graph associated with each satisfiable bound. Will save to the output directory or the current working directory if -o isn't specified.
