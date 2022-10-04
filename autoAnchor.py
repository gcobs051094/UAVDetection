import utils.autoanchor as autoAC

if __name__ == '__main__':
    new_anchors = autoAC.kmean_anchors('./data/DMS.yaml', 9, 640, 5.0, 1000, True)
    print(new_anchors)