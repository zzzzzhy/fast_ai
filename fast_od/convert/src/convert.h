void sample3d(size_t row_count, size_t column_count, size_t window_size, float *input, float *output);
void float32_convert(size_t _c, size_t _h, size_t _w, float *output, float *data);
void init_darknet(void);
char *calc_result(size_t orig_w, size_t orig_h, size_t shape, float *output);
