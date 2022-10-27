#pragma once
#ifndef ING_GEOMETRY_POINT_H_
#define ING_GEOMETRY_POINT_H_

/**
 * @file: include/ing/geometry/point.h
 * @author: sailing-innocent
 * @create: 2022-10-27
 * @desp: The Header for ING GEOMETRY Point
*/

#include <ing/geometry.h>

ING_NAMESPACE_BEGIN

class INGPoint: public INGGeoNode {
public:
    explicit INGPoint(size_t _dim = 1) { 
        mDim = _dim; 
        mCoord.resize(mDim);
        for (auto coord : mCoord) {
            coord = 0;
        }
    };
    virtual ~INGPoint() {}
    virtual size_t dim() const { return mDim; }
    virtual INGCoord& operator[](size_t index) { return mCoord[index]; }
    virtual std::vector<INGCoord>& vector() { return mCoord; }\
protected:
    size_t mDim = 0;
    std::vector<INGCoord> mCoord;
};

std::ostream &operator<<(std::ostream& os, INGPoint& p) {
    os << "point with " << p.dim() << " dims: ";
    os << "[";
    for (auto item : p.vector()) {
            os << item << ",";
    }
    os << "]" << std::endl;
    return os;
}

class INGPoint2D: public INGPoint {
public:
    explicit INGPoint2D(): INGPoint(2) {}
    explicit INGPoint2D(INGCoord x, INGCoord y)  { mDim = 2; mCoord.resize(2); mCoord[0] = x; mCoord[1] = y; }
    virtual ~INGPoint2D() {}
    virtual size_t dim() { return mDim; }
    virtual INGCoord& operator[](size_t index) {return mCoord[index]; }
    virtual std::vector<INGCoord>& vector(){ return mCoord; }
protected:
    size_t mDim = 2;
    std::vector<INGCoord> mCoord{0.0, 0.0};
};


class INGPoint3D: public INGPoint {
public:
    explicit INGPoint3D(): INGPoint(3) {}
    explicit INGPoint3D(INGCoord x, INGCoord y, INGCoord z)  { 
        mDim = 3; 
        mCoord.resize(3); 
        mCoord[0] = x; 
        mCoord[1] = y; 
        mCoord[2] = z;
    }
    virtual ~INGPoint3D() {}
    virtual size_t dim() { return mDim; }
    virtual INGCoord& operator[](size_t index) {return mCoord[index]; }
    virtual std::vector<INGCoord>& vector(){ return mCoord; }

protected:
    size_t mDim = 3;
    std::vector<INGCoord> mCoord{0.0, 0.0, 0.0};
};

class INGPointOrth: public INGPoint3D {
public:
    explicit INGPointOrth(): INGPoint3D() { mCoord.push_back(1.0);}
    explicit INGPointOrth(INGCoord x, INGCoord y, INGCoord z): INGPoint3D(x,y,z) {
        mCoord.push_back(1.0);
    }
    explicit INGPointOrth(INGPoint& p) {
        mDim = 3; mCoord.resize(4);
        if ( p.dim() == 2 ) {
            mCoord[0] = p[0];
            mCoord[1] = p[1];
            mCoord[2] = 0.0;
            mCoord[3] = 1.0;
        } else if ( p.dim() == 3 ) {
            mCoord[0] = p[0];
            mCoord[1] = p[1];
            mCoord[2] = p[2];
            mCoord[3] = 1.0;
        }
    }
    virtual ~INGPointOrth() {}
    virtual size_t dim() { return mDim; }
    virtual INGCoord& operator[](size_t index) {return mCoord[index]; }
    virtual std::vector<INGCoord>& vector(){ return mCoord; }
protected:
    size_t mDim = 3;
    std::vector<INGCoord> mCoord{0.0, 0.0, 0.0};
};

ING_NAMESPACE_END


#endif // ING_GEOMETRY_POINT_H_