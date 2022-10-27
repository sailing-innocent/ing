/**
 * @file: include/ing/model.h
 * @author: sailing-innocent
 * @create: 2022-10-27
 * @desp: The ING model Header
*/
#pragma once 
#ifndef ING_MODEL_H_
#define ING_MODEL_H_

/**
 * A Mathematical Model means it has
 * an input
 * innner structure
 * an output
*/

#include <ing/common.h>

ING_NAMESPACE_BEGIN

class INGModel: public INGNode {
public:
    INGModel() = default;
    virtual ~INGModel() {
        delete mInput;
        delete mOutput;
    }
protected:
    INGNode mInput;
    INGNode mOutput;
};

ING_NAMESPACE_END


#endif // ING_MODEL_H_
