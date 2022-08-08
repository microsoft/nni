import * as React from 'react';
import { DetailsRow, IDetailsRowBaseProps } from '@fluentui/react/lib/DetailsList';
import OpenRow from './OpenRow';
import '@style/table.scss';

interface ExpandableDetailsProps {
    detailsProps: IDetailsRowBaseProps;
    isExpand: boolean;
}

const ExpandableDetails = (props: ExpandableDetailsProps): any => {
    const { detailsProps, isExpand } = props;
    return (
        <div>
            <DetailsRow {...detailsProps} />
            {isExpand && <OpenRow trialId={detailsProps.item.id} />}
        </div>
    );
};

export default ExpandableDetails;
