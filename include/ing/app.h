/**
 * @file: include/ing/app.h
 * @author: sailing-innocent
 * @create: 2022-11-07
 * @desp: The Base INGApp Class
*/

#include <ing/common.h>

ING_NAMESPACE_BEGIN

class INGApp {
public:
    INGApp() = default;
    virtual ~INGApp() {}
public:
    virtual void init() = 0;
    virtual bool tick(int count) = 0;
    virtual void terminate() = 0;
};

ING_NAMESPACE_END