#pragma once
#ifndef ING_UTILS_COMMON_HPP_
#define ING_UTILS_COMMON_HPP_

/**
 * @file: include/utils/utils_common.hpp
 * @author: sailing-innocent
 * @create: 2022-11-06
 * @desp: The Common Utility Functions
*/

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ing/common.h>

ING_NAMESPACE_BEGIN

std::vector<char> readFile(const std::string& filename);

ING_NAMESPACE_END


#endif // ING_UTILS_COMMON_HPP_