#pragma once
#ifndef ING_GEOMETRY_H_
#define ING_GEOMETRY_H_

/**
 * @file: include/ing/geometry.h
 * @author: sailing-innocent
 * @create: 2022-10-27
 * @desp: The Entry for ING Geometry Part
*/


/**
 * For Drawing Purpose, the geometry library of ing will
 * consider few on mathematical definitions of geometries and will only use the 
 * most common lines and triangles 
 * */ 

#include <ing/common.h>
#include <vector>
#include <iostream>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


ING_NAMESPACE_BEGIN

ING_NAMESPACE_END

#endif // ING_GEOMETRY_H_