#pragma once 
#ifndef ING_GEOMETRY_SHAPE_H_
#define ING_GEOMETRY_SHAPE_H_

/**
 * @file: include/ing/geometry/shape.h
 * @author: sailing-innocent
 * @create: 2022-10-27
 * @desp: The Header for ING GEOMETRY Shape
*/

#include <ing/geometry.h>

ING_NAMESPACE_BEGIN

class INGShape: public INGGeoNode {
public:
    INGShape() = default;
    virtual ~INGShape() {}
};

// TODO: Create Triangle Shape With Color Attribe

ING_NAMESPACE_END

#endif // ING_GEOMETRY_SHAPE_H_
