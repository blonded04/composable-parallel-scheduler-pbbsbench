// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2010 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "common/time_loop.h"
#include "common/parse_command_line.h"
#include "common/sequenceIO.h"
#include <charconv>
#include <filesystem>
#include <iostream>
#include <algorithm>
using namespace std;
using namespace benchIO;

template <class T>
void timeIntegerSort(sequence<T> in_vals, int rounds, int bits, char* outFile) {
  size_t n = in_vals.size();
  sequence<T> R;
  time_loop(rounds, 1.0,
       [&] () {R.clear();},
       [&] () {R = int_sort(make_slice(in_vals.data(),in_vals.data()+n), bits);},
       [] () {});
  if (outFile != NULL) writeSequenceToFile(R, outFile);
}

std::string readFile(std::string filenameStr) {
  std::filesystem::path filename{filenameStr};
  auto fileSize = std::filesystem::file_size(filename);
  std::string content(fileSize, '\0');
  std::cout << "Now have string of size " << content.size() << std::endl;

  std::ifstream file{filenameStr};
  if (!file.read(content.data(), fileSize)) {
    std::cerr << "I CAN'T READ" << std::endl;
    throw std::runtime_error{"I CAN'T READ"};
  }

  return content;
}

struct Tokens {
  std::string content;
  std::vector<std::string_view> parts;
};

Tokens readTokens(std::string filenameStr) {
  Tokens tokensRes{.content = readFile(std::move(filenameStr))};

  std::string_view contentView = tokensRes.content;
  size_t lastNewLine = 0;
  for (auto newLine = contentView.find_first_of("\n "); newLine != std::string_view::npos; newLine = contentView.find_first_of("\n ", newLine + 1)) {
    size_t tokenStart = lastNewLine;
    size_t tokenSize = newLine - tokenStart;
    auto token = contentView.substr(tokenStart, tokenSize);
    tokensRes.parts.push_back(token);
    lastNewLine = newLine + 1;
  }
  return tokensRes;
}

template <class T>
T parseValue(std::string_view input) {
  T value{};
  auto res = std::from_chars(input.data(), input.data() + input.size(), value);
  if (res.ec != std::errc{}) {
    throw std::runtime_error{std::string{"couldn't parse value: "} + std::string{input}};
  }
  return value;
}

template <class T, class It>
sequence<T> parseValues(It begin, It end) {
  auto size = std::distance(begin, end);
  if constexpr (std::is_same_v<T, uint>){
    return parlay::tabulate(size, [&begin](size_t i) { return parseValue<uint>(*(begin + i)); });
  } else if constexpr (std::is_same_v<T, uintPair>) {
    return tabulate(size/2, [&] (long i) -> uintPair {
        return std::make_pair((uint) parseValue<uint>(*(begin + 2*i)), parseValue<uint>(*(begin + 2*i+1)));});
  } else {
    static_assert(std::is_same_v<T, uint>);
  }
}

int main(int argc, char* argv[]) {
  commandLine P(argc,argv,"[-o <outFile>] [-r <rounds>] <inFile>");
  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  int rounds = P.getOptionIntValue("-r",1);
  int bits = P.getOptionIntValue("-b",0);

  auto myIn = readTokens(iFile);
  elementType in_type = elementTypeFromHeader(myIn.parts[0]);
  cout << "bits = " << bits << endl;

  switch (in_type) {
  case intType: {
    auto inValues = parseValues<uint>(myIn.parts.begin() + 1, myIn.parts.end());
    std::cout << "Parsed " << inValues.size() << " uints" << std::endl;
    timeIntegerSort(std::move(inValues), rounds, bits, oFile);
    break;
  }
  case intPairT: {
    auto inValues = parseValues<uintPair>(myIn.parts.begin() + 1, myIn.parts.end());
    std::cout << "Parsed " << inValues.size() << " pairs" << std::endl;
    timeIntegerSort(std::move(inValues), rounds, bits, oFile);
    break;
  }
  default:
    cout << "integer Sort: input file not of right type" << endl;
    return(1);
  }
}

