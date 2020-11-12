#include "aps.h"
#include "ar.h"
#include "array.h"
#include "conf.h"
#include "file.h"
#include "perftest.h"
#include "test.h"
#include "types.h"

#include <exception>
#include <map>
#include <vector>

namespace aps {
namespace {

struct CommandLine {
	std::map<std::string, std::string> options;
	std::vector<std::string> arguments;
};
CommandLine readCommandLine(int argc, char* argv[]) {
	CommandLine cmdLine;
	int pos = 1;
	while(pos < argc && argv[pos][0] == '-') {
		std::string opt = argv[pos++];
		size_t dashCount = 0;
		while(dashCount < opt.size() && opt[dashCount] == '-') {
			++dashCount;
		}
		opt = opt.substr(dashCount);
		size_t eqPos = opt.find('=');
		if(eqPos == std::string::npos) {
			cmdLine.options[opt] = "";
		} else {
			cmdLine.options[opt.substr(0, eqPos)] = opt.substr(eqPos + 1);
		}
	}
	while(pos < argc) {
		cmdLine.arguments.push_back(argv[pos++]);
	}
	return cmdLine;
}

template <typename T>
struct TypeInfo {
	static const char* info() {
		return "";
	}
};
template <>
struct TypeInfo<LogDouble> {
	static const char* info() {
		return " (default; number is stored as its logarithm in a double)";
	}
};
template <>
struct TypeInfo<ExtDouble> {
	static const char* info() {
		return " (number is stored as a pair of normalized double and an 62-bit integer exponent)";
	}
};
template <>
struct TypeInfo<uint64_t> {
	static const char* info() {
		return " (exact integer computations modulo 2^64)";
	}
};

void printHelp(std::ostream& out) {
	out << "USAGE:\n";
	out << "  aps [OPTIONS] ordermodular INPUT_FILE OUTPUT_FILE\n";
	out << "    Solves APS in the order modular case\n";
	out << "\n";
	out << "  aps [OPTIONS] modular INPUT_FILE OUTPUT_FILE\n";
	out << "    Solves APS in the modular case (WARNING: might not be numerically stable)\n";
	out << "\n";
	out << "  aps [OPTIONS] ar_ordermodular INPUT_FILE OUTPUT_FILE\n";
	out << "    Solves AR in the order modular case\n";
	out << "\n";
	out << "  aps [OPTIONS] ar_modular INPUT_FILE OUTPUT_FILE\n";
	out << "    Solves AR in the modular case (WARNING: might not be numerically stable)\n";
	out << "\n";
	out << "  aps [OPTIONS] test\n";
	out << "    Runs tests, verifying that everything is working and the algorithms agree\n";
	out << "\n";
	out << "  aps [OPTIONS] perftest\n";
	out << "    Runs performance tests for each algorithm for inputs of different sizes\n";
	out << "\n";
	out << "  aps [OPTIONS] help\n";
	out << "    Shows this help\n";
	out << "\n";
	out << "OPTIONS:\n";
	out << "  -v, --verbose: Enable debug/progress output\n";
	out << "\n";
	out << "  --type=TYPE: Number type to use in the computations, available types:\n";
#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	out << "        \"" << #T << "\"" << TypeInfo<T>::info() << "\n";

	APS_FOR_EACH_NUMBER_TYPE
	out << "\n";

	out << "PROBLEM:\n";
	out << "  We have a set of variables, and for each variable and subset of other variables\n";
	out << "  (candidate parent set) we have a nonnegative weight.\n";
	out << "\n";
	out << "  In the modular case, the weight of a DAG is defined as the product of the parent\n";
	out << "  set weights of each variable.\n";
	out << "\n";
	out << "  In the order-modular case, the weight is further multiplied by the number of\n";
	out << "  linear extensions (topological orders) of the DAG.\n";
	out << "\n";
	out << "  In the APS (all parent sets) problem, we compute for each variable and candidate\n";
	out << "  parent set the total weight of DAGs where that variable has those parents.\n";
	out << "\n";
	out << "FILE FORMAT:\n";
	out << "  The format for both input and output files is the GOBNILP format:\n";
	out << "  <number of variables>\n";
	out << "  for each variable:\n";
	out << "    <variable name> <number of parent sets>\n";
	out << "    for each parent set:\n";
	out << "      <natural logarithm of weight> <number of parents> <names of parents separated by spaces>\n";
	out << "\n";
	out << "  In the AR (ancestor relations) problem, we compute for all pairs (i, j) of variables\n";
	out << "  the total weight of DAGs where j is an ancestor of i (a variable is always its own ancestor).\n";
	out << "  The input is in the same format as APS, and the output has the following format:\n";
	out << "\n";
	out << "AR OUTPUT FILE FORMAT:\n";
	out << "  <number of variables>\n";
	out << "  for each variable:\n";
	out << "    <variable name>\n";
	out << "  <square matrix: (row i, col j) = natural logarithm of total weight of DAGs where the jth variable\n";
	out << "                                   is an ancestor of the ith variable (-inf if there are none)>\n";
	out << "\n";
	out << "  The output always preserves the names and the ordering of the variables.\n";
	out << "\n";
}

template <typename T>
void solve(
	bool aps,
	bool orderModular,
	const std::string& inputFilename,
	const std::string& outputFilename
) {
	if(aps) {
		APSFuncList<T> funcs;
		if(orderModular) {
			funcs = getOrderModularAPSFuncs<T>();
		} else {
			funcs = getModularAPSFuncs<T>();
		}

		Instance<T> input = readInstance<T>(inputFilename);
		Instance<T> output;
		for(std::pair<std::string, APSFunc<T>> func : funcs) {
			output.weights = func.second(input.weights, false);
			if(output.weights.size() || !input.weights.size()) {
				break;
			}
		}
		if(input.weights.size() && !output.weights.size()) {
			fail("All methods rejected the instance (maybe it is too large?)");
		}
		output.names = move(input.names);
		writeInstance<T>(outputFilename, output);
	} else {
		ARFuncList<T> funcs;
		if(orderModular) {
			funcs = getOrderModularARFuncs<T>();
		} else {
			funcs = getModularARFuncs<T>();
		}

		Instance<T> input = readInstance<T>(inputFilename);
		AROutput<T> output;
		for(std::pair<std::string, APSFunc<T>> func : funcs) {
			output.weights = func.second(input.weights, false);
			if(output.weights.size() || !input.weights.size()) {
				break;
			}
		}
		if(input.weights.size() && !output.weights.size()) {
			fail("All methods rejected the instance (maybe it is too large?)");
		}
		output.names = move(input.names);
		writeAROutput(outputFilename, output);
	}
}

void run(int argc, char* argv[]) {
	try {
		CommandLine cmdLine = readCommandLine(argc, argv);

		if(cmdLine.options.count("v") || cmdLine.options.count("verbose")) {
			cmdLine.options.erase("v");
			cmdLine.options.erase("verbose");
			conf::verbose = true;
		}
		std::string numberType = "LogDouble";
		if(cmdLine.options.count("type")) {
			numberType = cmdLine.options["type"];
			cmdLine.options.erase("type");
		}

		void(*solveFunc)(bool, bool, const std::string&, const std::string&) = nullptr;

#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
		if(numberType == #T) { \
			solveFunc = solve<T>; \
		}

		APS_FOR_EACH_NUMBER_TYPE

		if(solveFunc == nullptr) {
			printHelp(std::cerr);
			fail("Unknown number type \"", numberType, "\"");
		}

		for(const std::pair<std::string, std::string>& opt : cmdLine.options) {
			printHelp(std::cerr);
			fail("Unknown option: ", opt.first);
		}

		if(cmdLine.arguments.size() == 3 && cmdLine.arguments[0] == "ordermodular") {
			solveFunc(true, true, cmdLine.arguments[1], cmdLine.arguments[2]);
			return;
		}
		if(cmdLine.arguments.size() == 3 && cmdLine.arguments[0] == "modular") {
			solveFunc(true, false, cmdLine.arguments[1], cmdLine.arguments[2]);
			return;
		}
		if(cmdLine.arguments.size() == 3 && cmdLine.arguments[0] == "ar_ordermodular") {
			solveFunc(false, true, cmdLine.arguments[1], cmdLine.arguments[2]);
			return;
		}
		if(cmdLine.arguments.size() == 3 && cmdLine.arguments[0] == "ar_modular") {
			solveFunc(false, false, cmdLine.arguments[1], cmdLine.arguments[2]);
			return;
		}
		if(cmdLine.arguments.size() == 1 && cmdLine.arguments[0] == "test") {
			runTests();
			return;
		}
		if(cmdLine.arguments.size() == 1 && cmdLine.arguments[0] == "perftest") {
			runPerfTests();
			return;
		}
		if(cmdLine.arguments.size() == 1 && cmdLine.arguments[0] == "help") {
			printHelp(std::cout);
			return;
		}

		printHelp(std::cerr);
		fail("Could not parse command line arguments");
	} catch(const std::exception& e) {
		fail("Unhandled exception: ", e.what());
	} catch(...) {
		fail("Unhandled unknown exception");
	}
}

}
}

int main(int argc, char* argv[]) {
	aps::run(argc, argv);
}
