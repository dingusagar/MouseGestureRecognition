from cfg import GestureRecognitionSettings
import time
import matplotlib.pyplot as plt

def get_logger(module_name, level=GestureRecognitionSettings.log_level):
    import logging
    logging.basicConfig()
    log = logging.getLogger(module_name)
    log.setLevel(level)
    return log


log = get_logger(__name__)




plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['ytick.major.left'] = False
plt.rcParams['ytick.major.right'] = False
plt.rcParams['ytick.minor.left'] = False
plt.rcParams['ytick.minor.left'] = False
plt.rcParams['xtick.major.top'] = False
plt.rcParams['xtick.major.bottom'] = False
plt.rcParams['xtick.minor.top'] = False
plt.rcParams['xtick.minor.bottom'] = False


def save_drawing(coordinates, directory=GestureRecognitionSettings.image_dir):
    X, Y = [cord[0] for cord in coordinates], [cord[1] for cord in coordinates]
    Y = [GestureRecognitionSettings.screen_res_y - y for y in Y] # inverting y cordinates on x-axis since the origin 0,0 starts at the topmost left of the screen.
    # matplotlib fig has origin at the bottom left corner.

    plt.clf()
    plt.plot(X, Y, color="#000", linewidth=5)
    filepath = directory / f'{time.time()}.png'
    log.debug(f'Saving image to {filepath}')
    plt.savefig(filepath)
    return filepath


if __name__ == "__main__":
    # testing
    coordinates = [(800, 531), (800, 530), (799, 530), (798, 530), (797, 529), (795, 528), (793, 527), (790, 526),
                   (787, 524), (782, 520), (776, 516), (767, 510), (758, 502), (748, 493), (740, 485), (732, 478),
                   (726, 469), (718, 461), (713, 454), (708, 447), (701, 437), (698, 429), (695, 423), (693, 416),
                   (692, 412), (692, 408), (692, 403), (692, 398), (692, 395), (692, 389), (692, 386), (692, 382),
                   (692, 377), (692, 372), (693, 366), (694, 361), (698, 354), (701, 349), (705, 343), (710, 337),
                   (716, 331), (723, 324), (731, 318), (739, 312), (747, 307), (755, 303), (764, 300), (773, 297),
                   (781, 296), (789, 294), (799, 293), (807, 294), (814, 294), (821, 297), (828, 300), (834, 304),
                   (841, 308), (849, 314), (857, 320), (864, 324), (871, 330), (879, 336), (887, 347), (895, 354),
                   (903, 364), (910, 372), (915, 382), (919, 389), (922, 398), (925, 407), (928, 416), (930, 425),
                   (931, 432), (931, 438), (931, 446), (931, 452), (931, 459), (929, 464), (929, 469), (928, 473),
                   (926, 480), (923, 486), (919, 492), (914, 496), (910, 500), (905, 502), (902, 504), (897, 507),
                   (894, 508), (888, 509), (883, 510), (877, 510), (872, 510), (866, 510), (860, 510), (855, 510),
                   (850, 510), (843, 510), (836, 510), (830, 509), (823, 509), (817, 508), (810, 506), (803, 506),
                   (799, 505), (794, 504), (789, 503), (786, 501), (781, 500), (779, 499), (776, 499), (774, 498),
                   (772, 497), (769, 496), (767, 495), (764, 494), (762, 493), (759, 492), (757, 491), (755, 490),
                   (754, 489), (752, 488), (752, 487), (751, 487), (750, 487), (749, 486), (748, 486), (748, 485)]
    GestureRecognitionSettings.image_dir.mkdir(exist_ok=True)
    save_drawing(coordinates)
