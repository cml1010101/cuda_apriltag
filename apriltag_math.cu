#include <apriltag_math.h>
__device__
void device_mat33_chol(const double* A, double* R)
{
    // A[0] = R[0]*R[0]
    R[0] = sqrt(A[0]);

    // A[1] = R[0]*R[3];
    R[3] = A[1] / R[0];

    // A[2] = R[0]*R[6];
    R[6] = A[2] / R[0];

    // A[4] = R[3]*R[3] + R[4]*R[4]
    R[4] = sqrt(A[4] - R[3]*R[3]);

    // A[5] = R[3]*R[6] + R[4]*R[7]
    R[7] = (A[5] - R[3]*R[6]) / R[4];

    // A[8] = R[6]*R[6] + R[7]*R[7] + R[8]*R[8]
    R[8] = sqrt(A[8] - R[6]*R[6] - R[7]*R[7]);

    R[1] = 0;
    R[2] = 0;
    R[5] = 0;
}
__global__
void kernel_mat33_chol(const double* A, double* R)
{
    device_mat33_chol(A, R);
}
void mat33_chol(const double* A, double* R)
{
    kernel_mat33_chol<<<1, 1>>>(A, R);
}
__device__
void device_mat33_lower_tri_inv(const double* A, double* R)
{
    // A[1] = R[0]*R[3];
    R[3] = A[1] / R[0];

    // A[2] = R[0]*R[6];
    R[6] = A[2] / R[0];

    // A[4] = R[3]*R[3] + R[4]*R[4]
    R[4] = sqrt(A[4] - R[3]*R[3]);

    // A[5] = R[3]*R[6] + R[4]*R[7]
    R[7] = (A[5] - R[3]*R[6]) / R[4];

    // A[8] = R[6]*R[6] + R[7]*R[7] + R[8]*R[8]
    R[8] = sqrt(A[8] - R[6]*R[6] - R[7]*R[7]);

    R[1] = 0;
    R[2] = 0;
    R[5] = 0;
}
__global__
void kernel_mat33_lower_tri_inv(const double* A, double* R)
{
    device_mat33_lower_tri_inv(A, R);
}
void mat33_lower_tri_inv(const double* A, double* R)
{
    kernel_mat33_lower_tri_inv<<<1, 1>>>(A, R);
}
__device__
void device_mat33_sym_solve(const double* A, const double* B, double* R)
{
    double L[9];
    device_mat33_chol(A, L);

    double M[9];
    device_mat33_lower_tri_inv(L, M);

    double tmp[3];
    tmp[0] = M[0]*B[0];
    tmp[1] = M[3]*B[0] + M[4]*B[1];
    tmp[2] = M[6]*B[0] + M[7]*B[1] + M[8]*B[2];

    R[0] = M[0]*tmp[0] + M[3]*tmp[1] + M[6]*tmp[2];
    R[1] = M[4]*tmp[1] + M[7]*tmp[2];
    R[2] = M[8]*tmp[2];
}
__global__
void kernel_mat33_sym_solve(const double* A, const double* B, double* R)
{
    device_mat33_sym_solve(A, B, R);
}
void mat33_sym_solve(const double* A, const double* B, double* R)
{
    kernel_mat33_sym_solve<<<1, 1>>>(A, B, R);
}