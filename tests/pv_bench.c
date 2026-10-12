/* SPDX-License-Identifier: MIT
 * Copyright (C) 2026 Andrew Youll */
/* One synthesis of the reference sentence (for counting instructions under
 * qemu, or timing on a real machine):  pv_bench VOICE REFDIR */
#include "pvoice.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	char err[200], path[512];
	int ids[2048], n = 0, x, ns;
	pv_voice *v;
	pv_options o;
	float *a;
	FILE *f;
	if (argc < 3)
		return 2;
	v = pv_load(argv[1], err, sizeof err);
	if (!v) {
		printf("%s\n", err);
		return 1;
	}
	snprintf(path, sizeof path, "%s/ids.txt", argv[2]);
	f = fopen(path, "r");
	while (f && n < 2048 && fscanf(f, "%d", &x) == 1)
		ids[n++] = x;
	if (f)
		fclose(f);
	pv_default_options(v, &o);
	o.seed = 1;
	if (argc > 3 && atoi(argv[3]) == 0) {   /* loading only */
		pv_free(v);
		return 0;
	}
	a = pv_synthesize(v, ids, n, &o, &ns, NULL);
	printf("%d samples\n", ns);
	free(a);
	pv_free(v);
	return 0;
}
