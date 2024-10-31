#define SIGNED_INT32_LIM 2147483648
#define UNSIGNED_INT32_LIM 4294967296

typedef float W;
typedef float X;
typedef short Y;

#ifdef _METAL
ptrdiff_t ptrdiff_t_min(ptrdiff_t a, ptrdiff_t b)
{
    if (a < b)
    {
        return a;
    }
    else
    {
        return b;
    }
}
#endif

