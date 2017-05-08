#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "option_list.h"



#include "utils.h"
#include "box.h"
#include "image.h"
#include "cuda.h"

#include "network.h"

#include "wrapper_c.h"

network net;
char** names;

void setup_network(char* cfgfile, char* weightfile){

    // TODO: this is hardcoded in the original code. It seems to be responsible for label namesChange it to allow custom config files
    char *datacfg = "res/coco.data";
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    names = get_labels(name_list);



    net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
}

void setup_cuda(){

    cuda_set_device(0);
}

void detect( char* filename, float thresh, int* hits,
    box** outboxes, float** outprobs, int** outclasses){

    float hier_thresh = 0.5;

    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms = .4;

    strncpy(input, filename, 256);

    image im = load_image_color(input, 0, 0);
    image sized = letterbox_image(im, net.w, net.h);
    //image sized = resize_image(im, net.w, net.h);
    //image sized2 = resize_max(im, net.w);
    //image sized = crop_image(sized2, -((net.w - sized2.w)/2), -((net.h - sized2.h)/2), net.w, net.h);
    //resize_network(&net, sized.w, sized.h);
    layer l = net.layers[net.n - 1];

    box *boxes = calloc(l.w * l.h * l.n, sizeof(box));
    float **probs = calloc(l.w * l.h * l.n, sizeof(float *));
    for (j = 0; j < l.w * l.h * l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

    float *X = sized.data;
    time = clock();
    network_predict(net, X);
    printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
    get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);

    printf("got the region boxes");

    if (nms) do_nms_obj(boxes, probs, l.w * l.h * l.n, l.classes, nms);
    //else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    printf("before the loop over boxes");

    int num = l.w * l.h * l.n;
    int classes = l.classes;

    int count = 0;

    for (int i = 0; i < num; ++i) {
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if (prob > thresh) {

            count++;
        }
    }

    printf("after loop over boxes. hits: %d", count);

    *hits=count;

    (*outboxes) = calloc(count, sizeof(box));
    (*outprobs) = calloc(count, sizeof(float));
    (*outclasses) = calloc(count, sizeof(int));

    int current_index=0;

    for (int i = 0; i < num; ++i) {
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if (prob > thresh) {

            printf("%s: %.0f%%\n", names[class], prob*100);

            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            box hb={b.x*im.w, b.y*im.h, b.w*im.w, b.h*im.h};

            (*outboxes)[current_index]=hb;
            (*outprobs)[current_index]=prob;
            (*outclasses)[current_index]=class;

            current_index++;
        }
    }

    printf("after final loop over boxes");


    free_image(im);
    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);
}

