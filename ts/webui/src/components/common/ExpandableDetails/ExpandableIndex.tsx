// interface ExpandableDetailsProps {
//     detailsProps: IDetailsRowBaseProps;
//     isExpand: boolean;
// }

import * as React from 'react';
import PropTypes from 'prop-types';
import { DetailsRow } from '@fluentui/react';
import OpenRow from './OpenRow';

const ExpandableDetails = (props): any => {
    const { detailsProps, isExpand } = props;
    return (
        <div>
            <DetailsRow {...detailsProps} />
            {isExpand && <OpenRow trialId={detailsProps.item.id} />}
        </div>
    );
};

ExpandableDetails.propTypes = {
    detailsProps: PropTypes.object,
    isExpand: PropTypes.bool
};

export default ExpandableDetails;
