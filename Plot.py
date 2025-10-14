import MyUtils
from Config import get_arguments



args = get_arguments()
MyUtils.load_and_plot_results("results", args.output_name)