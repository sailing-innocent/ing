#include <ing/geometry.h>
#include <iostream>

using namespace ing;

int main() {
    std::cout << "Hello" << std::endl;
    INGPoint2D p;
    std::cout << p.dim() << std::endl;
    std::cout << p[0] << " " << p[1] << std::endl;
    // change p
    p[0] = 1;
    std::cout << p[0] << std::endl;
    INGPoint2D p2{1,2};
    std::cout << p2.dim() << std::endl;
    std::cout << p2[0] << " " << p2[1] << std::endl;
    std::cout << "Hello 3D" << std::endl;
    INGPoint3D p3;
    std::cout << p3.dim() << std::endl;
    std::cout << p3[0] << " " << p3[1] << " " << p3[2] << std::endl;

    INGPoint3D p4{3.0, 4.0, 5.0};
    std::cout << p4.dim() << std::endl;
    std::cout << p4[0] << " " << p4[1] << " " << p4[2] << std::endl;

    std::cout << "Hello Orth" << std::endl;
    INGPointOrth p5;
    std::cout << p5.dim() << std::endl;
    std::cout << p5[0] << " " << p5[1] << " " << p5[2] << " " << p5[3] << std::endl;
    INGPointOrth p6(p2);
    std::cout << p6.dim() << std::endl;
    std::cout << p6[0] << " " << p6[1] << " " << p6[2] << " " << p6[3] << std::endl;
    std::cout << p6;
    return 0;
}
