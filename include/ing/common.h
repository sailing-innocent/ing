#pragma once
/**
 * @file: include/ing/common.h
 * @author: sailing-innocent
 * @create: 2022-10-15
 * @desp: The Common Definitions for ING
*/

#ifndef ING_COMMON_H_
#define ING_COMMON_H_

#define ING_NAMESPACE_BEGIN namespace ing {
#define ING_NAMESPACE_END }

ING_NAMESPACE_BEGIN

class Base {
public:
    Base() = default;
    /*
    Base(const Base&) = delete;
    Base(Base&&) = delete;
    Base& operator=(const Base&) = delete;
    Base& operator=(Base&&) = delete;
    */
    virtual ~Base() {}
};

class INGNode : public Base {
public:
    INGNode() = default;
    virtual ~INGNode() {};
};


ING_NAMESPACE_END

#endif // ING_COMMON_H_