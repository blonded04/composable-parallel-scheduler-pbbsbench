// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
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

#include <charconv>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "common/sequenceIO.h"
#include "common/atomics.h"
#include "common/parse_command_line.h"
using namespace std;
using namespace benchIO;

template <class T, class LESS>
void checkSort(sequence<T> in_vals,
	       sequence<T> out_vals,
	       LESS less) {
  size_t n = in_vals.size();
  // std::stable_sort(in_vals.begin(), in_vals.end(), less);
  // auto sorted_in = in_vals;
  auto sorted_in = parlay::stable_sort(in_vals, less);
  size_t error = n;
  parlay::parallel_for (0, n, [&] (size_t i) {
      if (out_vals[i] != sorted_in[i])
	pbbs::write_min(&error,i,std::less<size_t>());
  });
  if (error < n) {
    auto expected = parlay::to_chars(sorted_in[error]);
    auto got = parlay::to_chars(out_vals[error]);
    cout << "integer sort: check failed at location i=" << error
	 << " expected " << expected << " got " << got << endl;
    abort();
  }
}

std::string readFile(std::string filenameStr) {
  std::filesystem::path filename{filenameStr};
  auto fileSize = std::filesystem::file_size(filename);
  std::string content(fileSize, '\0');

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
  commandLine P(argc,argv,"<inFile> <outFile>");
  pair<char*,char*> fnames = P.IOFileNames();
  char* infile = fnames.first;
  char* outfile = fnames.second;
  
  auto myIn = readTokens(infile);
  elementType in_type = elementTypeFromHeader(myIn.parts[0]);
  size_t in_n = myIn.parts.size() - 1; // 126528

  auto myOut = readTokens(outfile);
  elementType out_type = elementTypeFromHeader(myOut.parts[0]);
  size_t out_n = myOut.parts.size() - 1; // 126528

  if (in_type != out_type) {
    cout << argv[0] << ": in and out types don't match" << endl;
    return(1);
  }
  
  if (in_n != out_n) {
    cout << argv[0] << ": in and out lengths don't match" << endl;
    return(1);
  }

  auto less = std::less<uint>{};
  auto lessp = [&] (uintPair a, uintPair b) {return a.first < b.first;};
  
  switch (in_type) {
  case intType: {
    auto inValues = parseValues<uint>(myIn.parts.begin() + 1, myIn.parts.end());
    auto outValues = parseValues<uint>(myOut.parts.begin() + 1, myOut.parts.end());
    checkSort(inValues, outValues, less);
    break;
  }
  case intPairT: {
    auto inValues = parseValues<uintPair>(myIn.parts.begin() + 1, myIn.parts.end());
    auto outValues = parseValues<uintPair>(myOut.parts.begin() + 1, myOut.parts.end());
    checkSort(inValues, outValues, lessp);
    break;
  }
  default:
    cout << argv[0] << ": input files not of right type" << endl;
    return(1);
  }
}
