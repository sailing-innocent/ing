#pragma once
#ifndef ING_GEOMETRY_H_
#define ING_GEOMETRY_H_

/**
 * @file: include/ing/geometry.h
 * @author: sailing-innocent
 * @create: 2022-10-27
 * @desp: The Entry for ING Geometry Part
*/

#include <ing/common.h>
#include <vector>
#include <iostream>

ING_NAMESPACE_BEGIN

typedef double INGCoord;

class INGGeoNode: public INGNode {
public:
    INGGeoNode() = default;
    virtual ~INGGeoNode() {}
    virtual size_t size() { return mSize; }
    virtual INGCoord& operator[](size_t index) { return mCoord[index]; }
protected:
    size_t mSize = 0;
    std::vector<INGCoord> mCoord;
};

ING_NAMESPACE_END

#include <ing/geometry/point.h>

#endif // ING_GEOMETRY_H_