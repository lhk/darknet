#ifndef WRAPPER_C
#define WRAPPER_C

void detect(char* cfgfile, char* weightfile, char* filename, float thresh, int* hits,
            box** outboxes, float** outprobs, int** outclasses);

void initialize();
#endif
