#!/bin/bash
#
# Build script for Self-Hosted AI Inference book
# Generates PDFs in ./gen folder
#
# Usage:
#   ./build.sh              # Build entire book
#   ./build.sh chapter01    # Build single chapter
#   ./build.sh all-chapters # Build all chapters individually
#   ./build.sh clean        # Clean generated files
#

set -e

# Configuration
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${ROOT_DIR}/src"
GEN_DIR="${ROOT_DIR}/gen"
LATEX_CMD="pdflatex"
MAKEINDEX_CMD="makeindex"

# Set TEXINPUTS to include our styles directory
export TEXINPUTS="${SRC_DIR}/styles:${SRC_DIR}:${TEXINPUTS}:"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create gen directory if it doesn't exist
mkdir -p "${GEN_DIR}"
mkdir -p "${GEN_DIR}/chapters"

# Function to print status
print_status() {
    echo -e "${GREEN}[BUILD]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to build the full book
build_full_book() {
    print_status "Building full book..."
    cd "${SRC_DIR}"

    # Run pdflatex twice for references, then makeindex, then pdflatex again
    ${LATEX_CMD} -output-directory="${GEN_DIR}" book.tex
    ${LATEX_CMD} -output-directory="${GEN_DIR}" book.tex

    # Make index if .idx file exists
    if [ -f "${GEN_DIR}/book.idx" ]; then
        cd "${GEN_DIR}"
        ${MAKEINDEX_CMD} -s "${SRC_DIR}/styles/svind.ist" book.idx
        cd "${SRC_DIR}"
        ${LATEX_CMD} -output-directory="${GEN_DIR}" book.tex
    fi

    print_status "Full book built: ${GEN_DIR}/book.pdf"
}

# Function to build a single chapter as standalone PDF
build_chapter() {
    local chapter_name=$1
    local chapter_file="chapters/${chapter_name}.tex"

    if [ ! -f "${SRC_DIR}/${chapter_file}" ]; then
        print_error "Chapter file not found: ${chapter_file}"
        return 1
    fi

    print_status "Building chapter: ${chapter_name}..."

    # Create a temporary standalone file for this chapter
    local temp_file="${GEN_DIR}/temp_${chapter_name}.tex"

    cat > "${temp_file}" << 'HEADER'
\documentclass[graybox,envcountchap,sectrefs]{svmono}

\usepackage{type1cm}
\usepackage{makeidx}
\usepackage{graphicx}
\usepackage{multicol}
\usepackage[bottom]{footmisc}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage[varvw]{newtxmath}
\usepackage{textcomp}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{upquote}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{positioning}

% Code listing style for Go
\lstdefinelanguage{Go}{
  keywords={break, case, chan, const, continue, default, defer, else, fallthrough, for, func, go, goto, if, import, interface, map, package, range, return, select, struct, switch, type, var},
  keywordstyle=\color{blue}\bfseries,
  ndkeywords={string, int, int64, float64, bool, error, nil, true, false, context, Context},
  ndkeywordstyle=\color{teal}\bfseries,
  identifierstyle=\color{black},
  sensitive=true,
  comment=[l]{//},
  morecomment=[s]{/*}{*/},
  commentstyle=\color{gray}\ttfamily,
  stringstyle=\color{red}\ttfamily,
  morestring=[b]',
  morestring=[b]",
  morestring=[b]`
}

% Code listing style - optimized for copy-paste
\lstset{
  basicstyle=\small\ttfamily,
  breakatwhitespace=false,
  breaklines=true,
  captionpos=b,
  keepspaces=true,
  columns=fullflexible,
  numbers=none,
  showspaces=false,
  showstringspaces=false,
  showtabs=false,
  tabsize=2,
  frame=single,
  framerule=0.5pt,
  rulecolor=\color{gray},
  backgroundcolor=\color{gray!5},
  upquote=true
}

\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  filecolor=magenta,
  urlcolor=cyan,
}

\graphicspath{{figures/}}

\makeindex

\begin{document}

\author{Author Name}
\title{Self-Hosted AI Inference}
\subtitle{A Systems Engineer's Guide}

\frontmatter
\maketitle
\tableofcontents

\mainmatter
HEADER

    # Add the chapter include
    echo "\\input{${chapter_file}}" >> "${temp_file}"

    # Close document
    echo "" >> "${temp_file}"
    echo "\\end{document}" >> "${temp_file}"

    # Build the chapter
    cd "${SRC_DIR}"
    ${LATEX_CMD} -output-directory="${GEN_DIR}" "${temp_file}" || true
    ${LATEX_CMD} -output-directory="${GEN_DIR}" "${temp_file}" || true

    # Move and rename output
    if [ -f "${GEN_DIR}/temp_${chapter_name}.pdf" ]; then
        mv "${GEN_DIR}/temp_${chapter_name}.pdf" "${GEN_DIR}/chapters/${chapter_name}.pdf"
        print_status "Chapter built: ${GEN_DIR}/chapters/${chapter_name}.pdf"
    else
        print_warning "PDF not generated for ${chapter_name} (may have LaTeX errors)"
    fi

    # Clean up temp files
    rm -f "${GEN_DIR}/temp_${chapter_name}".*
}

# Function to build all chapters individually
build_all_chapters() {
    print_status "Building all chapters individually..."

    for chapter_file in "${SRC_DIR}"/chapters/chapter*.tex; do
        if [ -f "$chapter_file" ]; then
            chapter_name=$(basename "$chapter_file" .tex)
            build_chapter "$chapter_name"
        fi
    done

    print_status "All chapters built in ${GEN_DIR}/chapters/"
}

# Function to build appendices
build_appendix() {
    local appendix_name=$1
    local appendix_file="appendices/${appendix_name}.tex"

    if [ ! -f "${SRC_DIR}/${appendix_file}" ]; then
        print_error "Appendix file not found: ${appendix_file}"
        return 1
    fi

    print_status "Building appendix: ${appendix_name}..."

    # Similar to chapter build but for appendices
    local temp_file="${GEN_DIR}/temp_${appendix_name}.tex"

    cat > "${temp_file}" << 'HEADER'
\documentclass[graybox,envcountchap,sectrefs]{svmono}

\usepackage{type1cm}
\usepackage{makeidx}
\usepackage{graphicx}
\usepackage{multicol}
\usepackage[bottom]{footmisc}
\usepackage{newtxtext}
\usepackage[varvw]{newtxmath}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{positioning}

\lstset{
  basicstyle=\small\ttfamily,
  breakatwhitespace=false,
  breaklines=true,
  captionpos=b,
  keepspaces=true,
  numbers=left,
  numbersep=5pt,
  numberstyle=\tiny\color{gray},
  showspaces=false,
  showstringspaces=false,
  showtabs=false,
  tabsize=2,
  frame=single,
  framerule=0.5pt,
  rulecolor=\color{gray},
  backgroundcolor=\color{gray!5}
}

\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  filecolor=magenta,
  urlcolor=cyan,
}

\graphicspath{{figures/}}

\makeindex

\begin{document}

\author{Author Name}
\title{Self-Hosted AI Inference}
\subtitle{A Systems Engineer's Guide}

\frontmatter
\maketitle
\tableofcontents

\mainmatter
HEADER

    echo "\\input{${appendix_file}}" >> "${temp_file}"
    echo "" >> "${temp_file}"
    echo "\\end{document}" >> "${temp_file}"

    cd "${SRC_DIR}"
    ${LATEX_CMD} -output-directory="${GEN_DIR}" "${temp_file}" || true
    ${LATEX_CMD} -output-directory="${GEN_DIR}" "${temp_file}" || true

    if [ -f "${GEN_DIR}/temp_${appendix_name}.pdf" ]; then
        mkdir -p "${GEN_DIR}/appendices"
        mv "${GEN_DIR}/temp_${appendix_name}.pdf" "${GEN_DIR}/appendices/${appendix_name}.pdf"
        print_status "Appendix built: ${GEN_DIR}/appendices/${appendix_name}.pdf"
    else
        print_warning "PDF not generated for ${appendix_name} (may have LaTeX errors)"
    fi

    rm -f "${GEN_DIR}/temp_${appendix_name}".*
}

# Function to build all appendices
build_all_appendices() {
    print_status "Building all appendices..."

    for appendix_file in "${SRC_DIR}"/appendices/appendix_*.tex; do
        if [ -f "$appendix_file" ]; then
            appendix_name=$(basename "$appendix_file" .tex)
            build_appendix "$appendix_name"
        fi
    done

    print_status "All appendices built in ${GEN_DIR}/appendices/"
}

# Function to clean generated files
clean() {
    print_status "Cleaning generated files..."
    rm -rf "${GEN_DIR}"
    print_status "Clean complete"
}

# Function to show help
show_help() {
    echo "Build script for Self-Hosted AI Inference book"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  (no args)       Build the entire book"
    echo "  book            Build the entire book"
    echo "  chapter01       Build a specific chapter (chapter01, chapter02, etc.)"
    echo "  all-chapters    Build all chapters as individual PDFs"
    echo "  appendix_a      Build a specific appendix"
    echo "  all-appendices  Build all appendices as individual PDFs"
    echo "  all             Build book + all individual chapters + all appendices"
    echo "  clean           Remove all generated files"
    echo "  help            Show this help message"
    echo ""
    echo "Output directory: ${GEN_DIR}"
}

# Main entry point
case "${1:-book}" in
    "book")
        build_full_book
        ;;
    "chapter"*)
        build_chapter "$1"
        ;;
    "all-chapters")
        build_all_chapters
        ;;
    "appendix"*)
        build_appendix "$1"
        ;;
    "all-appendices")
        build_all_appendices
        ;;
    "all")
        build_full_book
        build_all_chapters
        build_all_appendices
        ;;
    "clean")
        clean
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
