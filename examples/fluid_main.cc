#include <cstdio>
#include <array>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>

#include <thread>

using namespace std;
using namespace chrono;

#define MAX_STEP 1000
#define M_PI 3.141517

const size_t kBufferSize = 80;
const char* kGrayScaleTable = " .:-=+*#%@";
const size_t kGrayScaleTableSize = sizeof(kGrayScaleTable)/sizeof(char);

void updateWave(const double time_interval_in_s, double& x, double& speed_x) {

    x += time_interval_in_s * speed_x;

    // Boundary Change
    const double left_bound = 0.0;
    const double right_bound = 1.0; 
    if ( x > right_bound) {
        speed_x = - speed_x;
        x = 1.0 - time_interval_in_s * speed_x;
    }
    else if( x < left_bound) {
        speed_x = - speed_x;
        x = time_interval_in_s * speed_x;
    }
}

void accumulateWaveToHeightField(
    const double x,
    const double waveLength,
    const double waveHeight,
    array<double, kBufferSize>& height_field
) {
    const double quarterWaveLength = 0.25 * waveLength;
    const int start = static_cast<int>((x - quarterWaveLength)/1.0 * kBufferSize);
    const int end = static_cast<int>((x + quarterWaveLength)/1.0 * kBufferSize);

    for (int i = start; i < end; i++) {
        int iNew = i;
        if (i < 0) {
            iNew = -i - 1;
        } else if (i >= static_cast<int>(kBufferSize)) {
            iNew = 2 * kBufferSize - i - 1;
        }

        double distance = fabs(( i + 0.5)/kBufferSize - x);
        double height = waveHeight * 0.5 * cos(min(distance * M_PI / quarterWaveLength, M_PI)) + 1.0;
        height_field[iNew] += height;
    }

}

void draw(
    const array<double, kBufferSize>& heightField
) {
    string buffer(kBufferSize, ' ');

    // convert height_field to gray string
    for (size_t i = 0; i < kBufferSize; i++) {
        double height = heightField[i];
        size_t tableIndex = min(static_cast<size_t>(floor(height * kGrayScaleTableSize)), kGrayScaleTableSize - 1);
        buffer[i] = kGrayScaleTable[tableIndex];
    }

    // Clear older pointer
    for (size_t i = 0; i < kBufferSize; ++i) {
        printf("\b");
    }

    // Draw New Buffer
    printf("%s", buffer.c_str());
    fflush(stdout);
}

int main() {

    const double wavelength_x = 0.8;
    const double wavelength_y = 1.2;

    const double amplitude_x = 0.5;
    const double amplitude_y = 0.4;

    double x = 0.0;
    double y = 1.0;
    double speed_x = 2.0;
    double speed_y = -1.0;

    const int fps = 100;
    const double time_interval_in_s = 1.0 / fps; // in s

    array<double, kBufferSize> height_field;

    for (int i = 0; i < MAX_STEP; i++) {
        // Update wave
        updateWave(time_interval_in_s, x, speed_x);
        updateWave(time_interval_in_s, y, speed_y);

        // clear height field
        for (double& height: height_field) {
            height = 0.0;
        }

        // accumulate height field for each point
        accumulateWaveToHeightField(x, wavelength_x, amplitude_x, height_field);
        accumulateWaveToHeightField(y, wavelength_y, amplitude_y, height_field);

        draw(height_field);

        this_thread::sleep_for(milliseconds(1000/fps));
    }

    printf("\n");
    fflush(stdout);
    return 0;
}

