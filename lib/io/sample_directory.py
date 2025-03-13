# Fetch sample directories
DIR_NAMES = [('BC/benign_breast_cancer/v21', 'BC_benign'),
             ('BC/breast_cancer/v21', 'BC'),
             ('BL/bladder_cancer/v21', 'BL'),
             ('BT/biliary_tract_cancer/v21', 'BT'),
             ('CC/colorectal_cancer/v21', 'CC'),
             ('EC/esophageal_cancer/v21', 'EC'),
             ('GC/gastric_cancer/v21', 'GC'),
             ('GL/benign_skull/v21', 'GL_benign'),
             ('GL/glioma/v21', 'GL'),
             ('HC/v21', 'HC'),
             ('LK/v21', 'LK'),
             ('OV/benign_ovarian_cancer/v21', 'OV_benign'),
             ('OV/ovarian_cancer/v21', 'OV'),
             ('PC/v21', 'PC'),
             ('PR/benign_prostate_cancer/v21', 'PR_benign'),
             ('PR/prostate_cancer/v21', 'PR'),
             ('SA/v21/benign_primary', 'SA_benign'),
             ('SA/v21/malignant_primary', 'SA'),
             ('VOL/NCGG/v21', 'VOL'),
             ('VOL/minoru/XA/v21', 'VOL'),
             ('VOL/minoru/XB/v21', 'VOL')]


DIR_NAMES_comb = [('BC/benign_breast_cancer/v21', 'benign_neoplasm'),
             ('BC/breast_cancer/v21', 'BC'),
             ('BL/bladder_cancer/v21', 'BL'),
             ('BT/biliary_tract_cancer/v21', 'BT'),
             ('CC/colorectal_cancer/v21', 'CC'),
             ('EC/esophageal_cancer/v21', 'EC'),
             ('GC/gastric_cancer/v21', 'GC'),
             ('GL/benign_skull/v21', 'benign_neoplasm'),
             ('GL/glioma/v21', 'GL'),
             ('HC/v21', 'HC'),
             ('LK/v21', 'LK'),
             ('OV/benign_ovarian_cancer/v21', 'benign_neoplasm'),
             ('OV/ovarian_cancer/v21', 'OV'),
             ('PC/v21', 'PC'),
             ('PR/benign_prostate_cancer/v21', 'benign_neoplasm'),
             ('PR/prostate_cancer/v21', 'PR'),
             ('SA/v21/benign_primary', 'benign_neoplasm'),
             ('SA/v21/malignant_primary', 'SA'),
             ('VOL/NCGG/v21', 'VOL'),
             ('VOL/minoru/XA/v21', 'VOL'),
             ('VOL/minoru/XB/v21', 'VOL')]


DIR_NAMES_OUR = [('BC/BC', 'BC'),
             ('BL/BL', 'BL'),
             ('BT/BT', 'BT'),
             ('CC/CC', 'CC'),
             ('EC/EC', 'EC'),
             ('GC/GC', 'GC'),
             ('GL/GL', 'GL'),
             ('HC/HC', 'HC'),
             ('LK/LK', 'LK'),
             ('OV/OV', 'OV'),
             ('PC/PC', 'PC'),
             ('PR/PR', 'PR'),
             ('SA/SA', 'SA'),
             ('VOL/VOL', 'non_cancer')]


TEST_DIR_NAMES_TERNARY_LARGE = [('BL/bladder_cancer/v21', 'BL'),
                                ('GL/glioma/v21', 'GL'),
                                ('HC/v21', 'HC')]

TEST_DIR_NAMES_BINARY_LARGE = [('BL/bladder_cancer/v21', 'BL'),
                               ('GL/glioma/v21', 'GL')]

TEST_DIR_NAMES_BINARY_SMALL = [('BC/benign_breast_cancer/v21', 'BC_benign'),
                               ('GL/benign_skull/v21', 'GL_benign')]
